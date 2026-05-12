# 适配EEG信号解码的UniOT主文件
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pandas as pd
import ot
import os
import yaml
import torch.backends.cudnn as cudnn
from easydl import inverseDecaySheduler, OptimWithSheduler, TrainingModeManager, OptimizerManager, AccuracyCounter
from easydl import one_hot, variable_to_numpy, clear_output
from eval import eval as eval_func
from sklearn.model_selection import train_test_split

# ===================== 导入EEG相关模块和自定义组件 =====================
from utils.lib import seed_everything, sinkhorn, ubot_CCD, adaptive_filling
from utils.util import MemoryQueue
from utils.visualization import draw_tsne
from utils.net import ProtoCLS, CLS  # 保留原分类头/聚类头

# ===================== 基础配置 =====================
cudnn.benchmark = True
cudnn.deterministic = True

# 随机种子
seed = 1234
seed_everything(seed)

# 从YAML配置文件加载参数
config_file = 'UniOT-for-UniDA-main\config\eeg-config.yaml'
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✓ 配置已加载: {config_file}")
else:
    raise FileNotFoundError(f"配置文件 {config_file} 不存在！")

# 提取配置参数
source_classes = config['source_classes']
target_classes = config['target_classes']
temp = config['temp']
lam = config['lam']
mu = config['mu']
gamma = config['gamma']
K = config['K']
MQ_size = config['MQ_size']
batch_size = config['batch_size']
EEG_FEAT_DIM = config['EEG_FEAT_DIM']
feat_dim = config['feat_dim']

# GPU设置
gpu_index = config.get('gpu_index', '0')
if len(gpu_index) < 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    gpu_ids = list(map(int, gpu_index))
device = torch.device("cuda" if gpu_ids else "cpu")
print(f"✓ 设备已设置: {device}")

# 日志和配置保存
log_path = config['log']['root_dir']
os.makedirs(log_path, exist_ok=True)
log_dir = log_path
logger = SummaryWriter(log_dir)

# 保存配置
save_config = {"args": config, "seed": seed}
with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
    yaml.dump(save_config, f)
print(f"✓ 日志已保存到: {log_dir}")

# ===================== 核心参数配置 =====================
cls_output_dim = len(source_classes)

# ===================== 模型初始化（替换EEG特征提取器） =====================
# EEG特征提取器基础块
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 残差边（维度不匹配时调整）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 脑电专用轻量级 ResNet（输出 256 维特征）
class EEGResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channels=22, feat_dim=256):
        super().__init__()
        self.in_channels = 32
        # 初始卷积层：降采样并提取基础特征
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        
        # 三级残差网络：逐步增加通道数，减少时间维度
        self.layer1 = self._make_layer(32, 2)      # 保持分辨率
        self.layer2 = self._make_layer(64, 2, stride=2)  # 降采样
        self.layer3 = self._make_layer(128, 2, stride=2) # 降采样
        
        # 全局平均池化 + 特征映射到目标维度
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, feat_dim)

    def _make_layer(self, channels, num_blocks, stride=1):
        """构建残差层"""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock1D(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入: (batch, 22, 750) - 22通道EEG信号，750时间点
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.avgpool(x)      # (B, 128, 1)
        x = torch.flatten(x, 1)  # (B, 128)
        x = self.fc(x)           # (B, 256)
        return x  # 输出256维EEG特征向量

feature_extractor = EEGResNetFeatureExtractor(
    input_channels=22,  # EEG通道数（比如22通道）
    feat_dim=EEG_FEAT_DIM  # 输出特征维度
).to(device)

# 保留UniOT的分类头和聚类头（适配EEG特征维度）
classifier = CLS(
    in_dim=EEG_FEAT_DIM,  # 输入=EEG特征维度
    out_dim=cls_output_dim,
    hidden_mlp=512,      # 可根据EEG调整
    feat_dim=feat_dim,
    temp=temp
).to(device)

cluster_head = ProtoCLS(
    in_dim=feat_dim,
    out_dim=cls_output_dim,
    temp=temp
).to(device)

# ===================== 优化器配置 =====================
optimizer_featex = optim.SGD(
    feature_extractor.parameters(),
    lr=config['train']['lr']*0.1,
    weight_decay=float(config['train']['weight_decay']),
    momentum=config['train']['sgd_momentum'],
    nesterov=True
)
optimizer_cls = optim.SGD(
    classifier.parameters(),
    lr=config['train']['lr'],
    weight_decay=float(config['train']['weight_decay']),
    momentum=config['train']['sgd_momentum'],
    nesterov=True
)
optimizer_cluhead = optim.SGD(
    cluster_head.parameters(),
    lr=config['train']['lr'],
    weight_decay=float(config['train']['weight_decay']),
    momentum=config['train']['sgd_momentum'],
    nesterov=True
)

# 学习率衰减
scheduler = lambda step, initial_lr: inverseDecaySheduler(
    step, initial_lr, gamma=config['train'].get('lr_decay_gamma', 10), 
    power=config['train'].get('lr_decay_power', 0.75), 
    max_iter=config['train']['min_step']
)
opt_sche_featex = OptimWithSheduler(optimizer_featex, scheduler)
opt_sche_cls = OptimWithSheduler(optimizer_cls, scheduler)
opt_sche_cluhead = OptimWithSheduler(optimizer_cluhead, scheduler)

# ===================== 通用 EEG Dataset ======================
class EEGDataset(Dataset):
    """统一的EEG数据集类，支持源域和目标域"""
    def __init__(self, data_arr, label_arr):
        """
        Args:
            data_arr: (n_samples, channels, time_steps) - EEG信号数据
            label_arr: (n_samples,) - 对应标签
        """
        self.data = data_arr
        self.labels = label_arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回标准格式：(EEG信号, 标签, 样本ID)
        return self.data[idx], self.labels[idx], idx

# ===================== 数据加载与筛选 =====================
# 从配置加载文件路径并添加错误处理
source_file = config['data']['source_file']
target_file = config['data']['target_file']

try:
    source_npz = np.load(source_file)
    source_all_data = source_npz["data"]
    source_all_labels = source_npz["labels"]
    print(f"✓ 源域数据加载成功: {source_file}, 形状: {source_all_data.shape}")
    assert source_all_data.shape[1] == 22, f"源域数据通道数错误: {source_all_data.shape[1]} != 22"
except FileNotFoundError:
    raise FileNotFoundError(f"源域文件 {source_file} 不存在！")

mask = np.isin(source_all_labels, source_classes)
source_data = source_all_data[mask]
source_labels = source_all_labels[mask]
print(f"✓ 源域数据过滤完成: {len(source_labels)} 样本, 类别: {source_classes}")

# 目标域
try:
    target_npz = np.load(target_file)
    target_data = target_npz["data"]
    target_labels = target_npz["labels"]
    print(f"✓ 目标域数据加载成功: {target_file}, 形状: {target_data.shape}")
    assert target_data.shape[1] == 22, f"目标域数据通道数错误: {target_data.shape[1]} != 22"
except FileNotFoundError:
    raise FileNotFoundError(f"目标域文件 {target_file} 不存在！")

# ===================== 构建 Dataset & DataLoader =====================
# 源域：先拆分索引，再构建 Dataset（保证返回 3 个值：数据、标签、idx）
indices = np.arange(len(source_data))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=seed, stratify=source_labels
)

# 用索引拆分，保证返回 idx
source_train_ds = EEGDataset(source_data[train_idx], source_labels[train_idx])
source_val_ds = EEGDataset(source_data[val_idx], source_labels[val_idx])

# 目标域
target_train_data, target_test_data, target_train_labels, target_test_labels = train_test_split(
    target_data, target_labels, test_size=0.3, random_state=seed, stratify=target_labels
)
target_train_ds = EEGDataset(target_train_data, target_train_labels)
target_initMQ_ds = EEGDataset(target_train_data, target_train_labels)
target_test_ds = EEGDataset(target_test_data, target_test_labels)

# 构建 DataLoader（统一格式）
source_train_dl = DataLoader(source_train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
source_val_dl = DataLoader(source_val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
target_train_dl = DataLoader(target_train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
target_initMQ_dl = DataLoader(target_initMQ_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
target_test_dl = DataLoader(target_test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

# ===================== 内存队列初始化（适配EEG特征） =====================
target_size = len(target_train_ds)
n_batch = int(MQ_size / batch_size)
memqueue = MemoryQueue(feat_dim, batch_size, n_batch, temp).to(device)
cnt_i = 0

# 用目标域数据初始化内存队列
with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
    while cnt_i < n_batch:
        for i, (eeg_target, _, id_target) in enumerate(target_initMQ_dl):
            # EEG数据移到设备（EEG维度：[batch, channels, time_points]）
            current_batch_size = eeg_target.size(0)
            if current_batch_size != batch_size:
                continue
            eeg_target = eeg_target.to(device, dtype=torch.float32)
            id_target = id_target.to(device)
            
            # 提取EEG特征
            feature_ex = feature_extractor(eeg_target)
            before_lincls_feat, after_lincls = classifier(feature_ex)
            
            # 更新内存队列（归一化特征）
            memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
            cnt_i += 1
            if cnt_i > n_batch - 1:
                break

# ===================== UniOT核心训练循环（适配EEG） =====================
print("开始源域预训练...")
pretrain_epochs = config['train'].get('pretrain_epochs', 15)
best_acc = 0.0
criterion = nn.CrossEntropyLoss().to(device)
for epoch in range(pretrain_epochs):
    # 训练
    feature_extractor.train()
    classifier.train()
    for eeg_source, label_source, _ in source_train_dl:
        # 数据处理
        label_source = (label_source - 7).long().to(device, non_blocking=True)
        eeg_source = eeg_source.to(device, dtype=torch.float32, non_blocking=True)
        
        # 前向传播与损失计算
        feature_ex_s = feature_extractor(eeg_source)
        _, after_lincls_s = classifier(feature_ex_s)
        loss_cls = criterion(after_lincls_s, label_source)
        
        # 反向传播
        with OptimizerManager([opt_sche_featex, opt_sche_cls]):
            loss_cls.backward()
        
        # 权重归一化
        classifier.ProtoCLS.weight_norm()

    # 验证 - 简化准确率计算
    feature_extractor.eval()
    classifier.eval()
    val_correct = val_total = 0
    
    with torch.no_grad():
        for eeg_source_val, label_source_val, _ in source_val_dl:
            label_source_val = (label_source_val - 7).long().to(device, non_blocking=True)
            eeg_source_val = eeg_source_val.to(device, dtype=torch.float32, non_blocking=True)
            
            feature_ex_s_val = feature_extractor(eeg_source_val)
            _, after_lincls_s_val = classifier(feature_ex_s_val)
            _, predicted_val = torch.max(after_lincls_s_val, 1)
            
            val_total += label_source_val.size(0)
            val_correct += (predicted_val == label_source_val).sum().item()
    
    avg_val_acc = 100. * val_correct / val_total
    
    # 保存最优模型
    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        torch.save({
            'feature_extractor': feature_extractor.state_dict(),
            'classifier': classifier.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch + 1
        }, os.path.join(log_dir, 'best_eeg_uniot.pkl'))
    
    print(f"Epoch {epoch+1}/{pretrain_epochs} | Val Acc: {avg_val_acc:.2f}% | Best: {best_acc:.2f}%")
print("源域预训练完成！")

if hasattr(classifier, "module"):
    cluster_head.fc.weight.data = classifier.module.ProtoCLS.fc.weight.data.clone()
else:
    cluster_head.fc.weight.data = classifier.ProtoCLS.fc.weight.data.clone()
cluster_head.weight_norm()


total_steps = tqdm(range(config['train']['min_step']), desc='global step')
global_step = 0
beta = ot.unif(len(source_classes))  # 提前初始化beta

results = {}  # 初始化为字典而不是None

while global_step < config['train']['min_step']:
    iters = zip(source_train_dl, target_train_dl)
    for minibatch_id, ((eeg_source, label_source, id_source), (eeg_target, _, id_target)) in enumerate(iters):
        # ---------------------- 数据预处理 ----------------------
        label_source = (label_source - 7).long().to(device)
        eeg_source = eeg_source.to(device, dtype=torch.float32)
        eeg_target = eeg_target.to(device, dtype=torch.float32)
        id_target = id_target.to(device)

        # ---------------------- 特征提取（EEG） ----------------------
        # 源域EEG特征
        feature_ex_s = feature_extractor(eeg_source)
        before_lincls_feat_s, after_lincls_s = classifier(feature_ex_s)
        norm_feat_s = F.normalize(before_lincls_feat_s)

        # 目标域EEG特征
        feature_ex_t = feature_extractor(eeg_target)
        before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)
        norm_feat_t = F.normalize(before_lincls_feat_t)

        # 目标域聚类头输出
        after_cluhead_t = cluster_head(before_lincls_feat_t)

        # ---------------------- 1. 源域监督损失 ----------------------
        loss_cls = criterion(after_lincls_s, label_source)

        # ---------------------- 2. 私有类别发现（PCD） ----------------------
        minibatch_size = norm_feat_t.size(0)

        # 计算目标域样本间相似度（排除自身）
        feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / temp
        mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(device)
        feat_mat2.masked_fill_(mask, -1/temp)

        # 从内存队列找最近邻
        nb_value_tt, nb_feat_tt = memqueue.get_nearest_neighbor(norm_feat_t, id_target)
        neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
        values, indices = torch.max(neighbor_candidate_sim, 1)
        
        # 构建近邻特征
        neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).to(device)
        for i in range(minibatch_size):
            neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), norm_feat_t], 0)
            neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]
        neighbor_output = cluster_head(neighbor_norm_feat)

        # 内存队列填充特征
        fill_size_ot = K
        mqfill_feat_t = memqueue.random_sample(fill_size_ot)
        mqfill_output_t = cluster_head(mqfill_feat_t)

        # OT过程（Sinkhorn迭代）
        S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
        S_tt *= temp
        Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=30)
        Q_tt_tilde = Q_tt * Q_tt.size(0)
        anchor_Q = Q_tt_tilde[:minibatch_size, :]
        neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]

        # 计算PCD损失
        loss_local = 0
        for i in range(minibatch_size):
            sub_loss_local = -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:]))
            sub_loss_local += -torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:]))
            sub_loss_local /= 2
            loss_local += sub_loss_local
        loss_local /= minibatch_size
        loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
        loss_PCD = (loss_global + loss_local) / 2

        # ---------------------- 3. 公共类别检测（CCD） ----------------------
        loss_CCD = 0.0
        if global_step > 100:  # 预热后启动CCD
            # 源域原型（分类头权重）
            source_prototype = classifier.ProtoCLS.fc.weight if hasattr(classifier, "module") else classifier.ProtoCLS.fc.weight
            if beta is None:
                beta = ot.unif(source_prototype.size()[0])

            # 内存队列填充特征
            fill_size_uot = n_batch * batch_size
            mqfill_feat_t = memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
            
            # 自适应填充
            newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, gamma, beta, fill_size_uot)
        
            # UOT-based CCD
            high_conf_label_id, high_conf_label, _, new_beta = ubot_CCD(
                newsim, beta, fake_size=fake_size, fill_size=fill_size_uot, mode='minibatch'
            )
            beta = mu * beta + (1 - mu) * new_beta

            # 计算CCD损失（避免空样本）
            if high_conf_label_id.size(0) > 0:
                loss_CCD = criterion(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])

        # ---------------------- 总损失 & 反向传播 ----------------------
        loss_all = loss_cls + lam * (loss_PCD + loss_CCD)
        
        with OptimizerManager([opt_sche_featex, opt_sche_cls, opt_sche_cluhead]):
            loss_all.backward()

        # 原型分类头权重归一化（UniOT关键操作）
        if hasattr(classifier, "module"):
            classifier.ProtoCLS.weight_norm()
            cluster_head.weight_norm()
        else:
            classifier.ProtoCLS.weight_norm()
            cluster_head.weight_norm()

        # 更新内存队列
        current_bs = norm_feat_t.size(0)
        if current_bs == batch_size:
            memqueue.update_queue(norm_feat_t, id_target)
        global_step += 1
        total_steps.update()

        # ---------------------- 日志记录 ----------------------
        if global_step % config['log']['log_interval'] == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), 
                               variable_to_numpy(after_lincls_s))
            acc_source = torch.tensor([counter.reportAccuracy()]).to(device)
            logger.add_scalar('loss_all', loss_all, global_step)
            logger.add_scalar('loss_cls', loss_cls, global_step)
            logger.add_scalar('loss_PCD', loss_PCD, global_step)
            logger.add_scalar('loss_CCD', loss_CCD, global_step)
            logger.add_scalar('acc_source', acc_source, global_step)

        # ---------------------- 测试评估 ----------------------
        if global_step % config['test']['test_interval'] == 0:
            # 准备完整的classes_set
            common_classes = list(set(source_classes) & set(target_classes))
            tp_classes = [c for c in target_classes if c not in source_classes]
            classes_set = {
                "source_classes": source_classes,
                "target_classes": target_classes,
                "common_classes": common_classes,
                "tp_classes": tp_classes
            }
            try:
                results = eval_func(
                    feature_extractor, classifier, target_test_dl, 
                    classes_set,
                    gamma=gamma, beta=beta, seed=seed
                )
            except Exception as e:
                print(f"⚠ 评估失败: {e}")
                traceback.print_exc()  # 打印完整报错堆栈
                continue
            # 记录EEG解码指标
            if results:
                logger.add_scalar('cls_common_acc', results['cls_common_acc'], global_step)
                logger.add_scalar('cls_tp_acc', results['cls_tp_acc'], global_step)
                logger.add_scalar('tp_nmi', results['tp_nmi'], global_step)
                logger.add_scalar('cls_overall_acc', results['cls_overall_acc'], global_step)
                logger.add_scalar('h_score', results['h_score'], global_step)
            clear_output()

        if global_step >= config['train']['min_step']:
            break

# ===================== 模型保存 & 结果导出 =====================
# 保存最终模型
save_dict = {
    "feature_extractor": feature_extractor.state_dict(),
    "classifier": classifier.state_dict(),
    "cluster_head": cluster_head.state_dict(),
    "K": K,
    "beta": beta,  # 直接保存numpy array，无需转换
    "source_classes": source_classes,
    "target_classes": target_classes
}
torch.save(save_dict, os.path.join(log_dir, 'final_eeg_uniot.pkl'))
print(f"✓ 模型已保存: {os.path.join(log_dir, 'final_eeg_uniot.pkl')}")

# 保存测试结果（检查非空）
if results:
    pd.DataFrame([results]).to_csv(f'{log_dir}/eeg_result.csv')
    print(f"✓ 结果已保存: {log_dir}/eeg_result.csv")
else:
    print("⚠ 未进行过评估，无结果保存")

# EEG特征可视化（可选）
if config['dataset'] in ['eeg_dataset']:
    writer = SummaryWriter(f'{log_path}/tsne')
    draw_tsne(
        feature_extractor, classifier, cluster_head,
        source_train_dl, target_test_dl,
        source_classes, target_classes,
        writer, config['dataset']
    )