import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pandas as pd
import ot
import os
import datetime
from utils.lib import seed_everything, sinkhorn, ubot_CCD, adaptive_filling
from utils.util import MemoryQueue
from utils.net import CLS, ProtoCLS
from eval import eval  # Reuse eval function
from easydl import inverseDecaySheduler, OptimWithSheduler, TrainingModeManager, OptimizerManager, AccuracyCounter
from easydl import one_hot, variable_to_numpy

# Set seed
seed = 1234
seed_everything(seed)

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Argument parser（完全保留）
parser = argparse.ArgumentParser(description='EEG UniOT Training')
parser.add_argument('--gpu_index', type=str, default='0', help='GPU index')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--MQ_size', type=int, default=2000, help='Memory queue size')
parser.add_argument('--K', type=int, default=50, help='Number of clusters')
parser.add_argument('--gamma', type=float, default=0.7, help='Gamma for adaptive filling')
parser.add_argument('--mu', type=float, default=0.7, help='Mu for beta update')
parser.add_argument('--temp', type=float, default=0.1, help='Temperature')
parser.add_argument('--lam', type=float, default=0.1, help='Lambda for loss weighting')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--min_step', type=int, default=10000, help='Minimum training steps')
parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
parser.add_argument('--test_interval', type=int, default=500, help='Test interval')
parser.add_argument('--output_dir', type=str, default='./log/eeg_ubot', help='Output directory')
parser.add_argument('--feature_extractor_path', type=str, default=None, help='Path to pretrained feature extractor')
args = parser.parse_args()

# GPU setup
if args.gpu_index:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

# Output directory
os.makedirs(args.output_dir, exist_ok=True)
log_dir = args.output_dir
logger = SummaryWriter(log_dir)

# EEG Dataset
class EEGDataset(Dataset):
    def __init__(self, arrays, labels):
        self.arrays = arrays
        self.labels = labels

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        arr = self.arrays[idx]
        label = self.labels[idx]
        return arr, label, idx

# EEG Feature Extractor
class EEGFeatureExtractor(nn.Module):
    def __init__(self, input_channels=22, time_steps=750, output_dim=2048):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# ===================== 核心修改：只保留标签 7、8、9 作为源域 =====================
# 加载源域数据
source_npz = np.load(r"D:\vscode_code\UniOT-for-UniDA-main\UniOT-for-UniDA-main\A01T_A08T.npz")
source_all_data = source_npz["data"]
source_all_labels = source_npz["labels"]

# 筛选：只保留标签 7、8、9
mask = np.isin(source_all_labels, [7, 8, 9])
source_arrays = source_all_data[mask]
source_labels = source_all_labels[mask]

# 加载目标域（不变）
target_npz = np.load(r"D:\vscode_code\UniOT-for-UniDA-main\UniOT-for-UniDA-main\A09T.npz")
target_arrays = target_npz["data"]
target_labels = target_npz["labels"]
# ================================================================================

# Assume classes are partitioned (simplified: all classes are common for demo)
n_classes = len(np.unique(source_labels))
source_classes = list(range(n_classes))
target_classes = list(range(n_classes))
classes_set = {
    'source_classes': source_classes,
    'target_classes': target_classes,
    'tp_classes': [],
    'sp_classes': [],
    'common_classes': source_classes
}

# Transforms (simple normalization)
def eeg_transform(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

source_ds = EEGDataset(source_arrays, source_labels, transform=eeg_transform)
target_ds = EEGDataset(target_arrays, target_labels if target_labels is not None else np.zeros(len(target_arrays)), transform=eeg_transform)

source_dl = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
target_dl = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
target_initMQ_dl = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

# Model setup
feat_dim = 256
feature_extractor = EEGFeatureExtractor()
if args.feature_extractor_path:
    feature_extractor.load_state_dict(torch.load(args.feature_extractor_path))
classifier = CLS(feature_extractor.output_dim, len(source_classes), hidden_mlp=2048, feat_dim=feat_dim, temp=args.temp)
cluster_head = ProtoCLS(feat_dim, args.K, temp=args.temp)

feature_extractor = feature_extractor.cuda()
classifier = classifier.cuda()
cluster_head = cluster_head.cuda()

optimizer_featex = optim.SGD(feature_extractor.parameters(), lr=args.lr*0.1, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)
optimizer_cls = optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)
optimizer_cluhead = optim.SGD(cluster_head.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=args.min_step)
opt_sche_featex = OptimWithSheduler(optimizer_featex, scheduler)
opt_sche_cls = OptimWithSheduler(optimizer_cls, scheduler)
opt_sche_cluhead = OptimWithSheduler(optimizer_cluhead, scheduler)

feature_extractor = nn.DataParallel(feature_extractor).train(True)
classifier = nn.DataParallel(classifier).train(True)
cluster_head = nn.DataParallel(cluster_head).train(True)

# Memory queue init
target_size = len(target_ds)
n_batch = int(args.MQ_size / args.batch_size)
memqueue = MemoryQueue(feat_dim, args.batch_size, n_batch, args.temp).cuda()
cnt_i = 0
with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
    while cnt_i < n_batch:
        for i, (im_target, _, id_target) in enumerate(target_initMQ_dl):
            im_target = im_target.cuda()
            id_target = id_target.cuda()
            feature_ex = feature_extractor(im_target)
            before_lincls_feat, after_lincls = classifier(feature_ex)
            memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
            cnt_i += 1
            if cnt_i >= n_batch:
                break

# Training
total_steps = tqdm(range(args.min_step), desc='global step')
global_step = 0
beta = None
best_loss = float('inf')
results_history = []

while global_step < args.min_step:
    iters = zip(source_dl, target_dl)
    for minibatch_id, ((im_source, label_source, id_source), (im_target, _, id_target)) in enumerate(iters):
        label_source = label_source.cuda()
        im_source = im_source.cuda()
        im_target = im_target.cuda()

        feature_ex_s = feature_extractor(im_source)
        feature_ex_t = feature_extractor(im_target)

        before_lincls_feat_s, after_lincls_s = classifier(feature_ex_s)
        before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)

        norm_feat_s = F.normalize(before_lincls_feat_s)
        norm_feat_t = F.normalize(before_lincls_feat_t)

        after_cluhead_t = cluster_head(before_lincls_feat_t)

        # Source Supervision
        criterion = nn.CrossEntropyLoss().cuda()
        loss_cls = criterion(after_lincls_s, label_source)

        # Private Class Discovery
        minibatch_size = norm_feat_t.size(0)
        feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / args.temp
        mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().cuda()
        feat_mat2.masked_fill_(mask, -1/args.temp)

        nb_value_tt, nb_feat_tt = memqueue.get_nearest_neighbor(norm_feat_t, id_target.cuda())
        neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
        values, indices = torch.max(neighbor_candidate_sim, 1)
        neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).cuda()
        for i in range(minibatch_size):
            neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), norm_feat_t], 0)
            neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]
            
        neighbor_output = cluster_head(neighbor_norm_feat)
        
        fill_size_ot = args.K
        mqfill_feat_t = memqueue.random_sample(fill_size_ot)
        mqfill_output_t = cluster_head(mqfill_feat_t)

        S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
        S_tt *= args.temp
        Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
        Q_tt_tilde = Q_tt * Q_tt.size(0)
        anchor_Q = Q_tt_tilde[:minibatch_size, :]
        neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]

        loss_local = 0
        for i in range(minibatch_size):
            sub_loss_local = -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:])) - torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:]))
            sub_loss_local /= 2
            loss_local += sub_loss_local
        loss_local /= minibatch_size
        loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
        loss_PCD = (loss_global + loss_local) / 2

        # Common Class Detection
        if global_step > 100:
            source_prototype = classifier.module.ProtoCLS.fc.weight
            if beta is None:
                beta = ot.unif(source_prototype.size()[0])

            fill_size_uot = n_batch * args.batch_size
            mqfill_feat_t = memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
            
            newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, args.gamma, beta, fill_size_uot)
        
            high_conf_label_id, high_conf_label, _, new_beta = ubot_CCD(newsim, beta, fake_size=fake_size, fill_size=fill_size_uot, mode='minibatch')
            beta = args.mu * beta + (1 - args.mu) * new_beta

            if high_conf_label_id.size(0) > 0:
                loss_CCD = criterion(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])
            else:
                loss_CCD = 0
        else:
            loss_CCD = 0
        
        loss_all = loss_cls + args.lam * (loss_PCD + loss_CCD)
        
        with OptimizerManager([opt_sche_featex, opt_sche_cls, opt_sche_cluhead]):
            loss_all.backward()

        classifier.module.ProtoCLS.weight_norm()
        cluster_head.module.weight_norm()
        memqueue.update_queue(norm_feat_t, id_target.cuda())
        global_step += 1
        total_steps.update()
        
        if global_step % args.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(after_lincls_s))
            acc_source = torch.tensor([counter.reportAccuracy()]).cuda()
            logger.add_scalar('loss_all', loss_all, global_step)
            logger.add_scalar('loss_cls', loss_cls, global_step)
            logger.add_scalar('loss_PCD', loss_PCD, global_step)
            logger.add_scalar('loss_CCD', loss_CCD, global_step)
            logger.add_scalar('acc_source', acc_source, global_step)

        if global_step % args.test_interval == 0:
            results = eval(feature_extractor, classifier, target_dl, classes_set, gamma=args.gamma, beta=beta)
            logger.add_scalar('cls_common_acc', results['cls_common_acc'], global_step)
            logger.add_scalar('cls_tp_acc', results['cls_tp_acc'], global_step)
            logger.add_scalar('tp_nmi', results['tp_nmi'], global_step)
            logger.add_scalar('cls_overall_acc', results['cls_overall_acc'], global_step)
            logger.add_scalar('h_score', results['h_score'], global_step)
            logger.add_scalar('h3_score', results['h3_score'], global_step)
            
            # Record metrics
            results_history.append({
                'step': global_step,
                'common_acc': results['cls_common_acc'],
                'h_score': results['h_score']
            })
            
            # Save best model based on loss
            if loss_all < best_loss:
                best_loss = loss_all.item()
                data = {
                    "feature_extractor": feature_extractor.state_dict(),
                    "classifier": classifier.state_dict(),
                    'cluster_head': cluster_head.state_dict(),
                    'K': args.K,
                    'beta': torch.from_numpy(beta)
                }
                torch.save(data, os.path.join(log_dir, 'best.pkl'))

# Save final model
data = {
    "feature_extractor": feature_extractor.state_dict(),
    "classifier": classifier.state_dict(),
    'cluster_head': cluster_head.state_dict(),
    'K': args.K,
    'beta': torch.from_numpy(beta)
}
torch.save(data, os.path.join(log_dir, 'final.pkl'))

# Save results history
pd.DataFrame(results_history).to_csv(os.path.join(log_dir, 'results_history.csv'), index=False)