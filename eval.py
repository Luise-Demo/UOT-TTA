from easydl import variable_to_numpy
from easydl import TrainingModeManager, Accumulator
import numpy as np
from utils.lib import seed_everything, ubot_CCD, adaptive_filling
from utils.util import ResultsCalculator
import torch
import torch.nn.functional as F
from tqdm import tqdm
import ot
from sklearn.cluster import KMeans
import os

def run_kmeans(L2_feat, ncentroids, init_centroids=None, seed=None, gpu=False, min_points_per_centroid=1):
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    
    if torch.is_tensor(L2_feat):
        L2_feat = variable_to_numpy(L2_feat)
    
    # Use sklearn KMeans instead of faiss
    kmeans = KMeans(n_clusters=ncentroids, 
                    random_state=seed, 
                    n_init=5,  # equivalent to nredo=5
                    max_iter=20,  # equivalent to niter=20
                    init=init_centroids if init_centroids is not None else 'k-means++')
    kmeans.fit(L2_feat)
    
    pred_centroid = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    return pred_centroid, centroids


def eval(feature_extractor, classifier, eval_dl, classes_set, 
        gamma=0.7, beta=None, seed=None, uniformed_index=None):
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    if uniformed_index is None:
        uniformed_index = len(classes_set['source_classes'])
    if beta is None:
        beta = ot.unif(source_prototype.size()[0])
    
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
            Accumulator(['label_t', 'norm_feat_t']) as eval_accumulator, \
            torch.no_grad():
        for i, (im_t, label_t, *_) in enumerate(tqdm(eval_dl, desc='testing')):
            im_t = im_t.cuda()
            label_t = label_t.cuda()
            feature_ex_t = feature_extractor.forward(im_t)
            before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            val = dict()
            for name in eval_accumulator.names:
                val[name] = locals()[name].cpu().data.numpy()
            eval_accumulator.updateData(val)  

    for x in eval_accumulator:
        val[x] = eval_accumulator[x] 
    label_t = val['label_t']
    norm_feat_t = val['norm_feat_t']
    del val

    # Unbalanced OT
    source_prototype = classifier.ProtoCLS.fc.weight

    stopThr = 1e-6
    # Adaptive filling 
    newsim, fake_size = adaptive_filling(torch.from_numpy(norm_feat_t).cuda(), 
                                        source_prototype, gamma, beta, 0, stopThr=stopThr)

    # obtain predict label
    _, __, pred_label, ___ = ubot_CCD(newsim, beta, fake_size=fake_size, fill_size=0, mode='minibatch', stopThr=stopThr)
    pred_label = pred_label.cpu().data.numpy()

    # obtain private samples
    filter = (lambda x: x in classes_set["tp_classes"])
    private_mask = np.zeros((label_t.size,), dtype=bool) 
    for i in range(label_t.size):
        if filter(label_t[i]):
            private_mask[i] = True
    private_feat = norm_feat_t[private_mask, :]
    private_label = label_t[private_mask]

    # obtain results
    ncentroids = len(classes_set["tp_classes"])
    private_pred, _ = run_kmeans(private_feat, ncentroids, init_centroids=None, seed=seed, gpu=True)
    results = ResultsCalculator(classes_set, label_t, pred_label, private_label, private_pred)
    results_dict = {
        'cls_common_acc': results.common_acc_aver,
        'cls_tp_acc': results.tp_acc,
        'tp_nmi': results.tp_nmi,
        'cls_overall_acc': results.overall_acc_aver,
        'h_score': results.h_score,
        'h3_score': results.h3_score
    }
    return results_dict


if __name__ == '__main__':
    from data import *
    from utils.net import ResNet50Fc, CLS
    import torch.nn as nn

    if parser_args.model_path is None:
        raise ValueError('NO model_path input!')

    seed = 1234
    seed_everything(seed)

    if len(parser_args.gpu_index) < 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu_ids = []
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu_index
        gpu_ids = list(map(int, parser_args.gpu_index))

    # define network architecture
    cls_output_dim = len(source_classes)
    feat_dim = 256
    feature_extractor = ResNet50Fc(pretrained_model_path)
    classifier = CLS(feature_extractor.output_dim, cls_output_dim, hidden_mlp=2048, feat_dim=256, temp=temp)

    # to cuda
    feature_extractor = feature_extractor.cuda()
    classifier = classifier.cuda()

    # DataParallel
    feature_extractor = nn.DataParallel(feature_extractor).train(False)
    classifier = nn.DataParallel(classifier).train(False)

    # load model
    data = torch.load(open(parser_args.model_path, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    beta = data['beta']

    results = eval(feature_extractor, classifier, target_test_dl, classes_set, gamma=gamma, beta=beta)

    print(results)


