import torch
import dgl
import networkx as nx
import random
import math
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from sklearn.manifold import TSNE
from dgl.data import CoraGraphDataset, CiteseerGraphDataset
import copy

def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def load_dgl_cite_dataset(name, dir='/xyzhao/datasets', local=True):
    if name == 'cora':
        if local:
            data = dgl.data.CoraGraphDataset(raw_dir=dir, force_reload=False)
        else:
            data = CoraGraphDataset()
    elif name == 'citeseer':
        if local:
            data = dgl.data.CiteseerGraphDataset(raw_dir=dir, force_reload=False)
        else:
            data = CiteseerGraphDataset()
    else:
        raise ValueError("Unsupported dataset name {}".format(name))
    return data

def select_and_inject_nodes(g, frac_n, frac_f):
    # only support dgl graph for now, not very efficient for now
    # add a node to each selected nodes

    n_nodes = g.num_nodes()
    n_feats = g.ndata['feat'].shape[-1]
    sel_nids = random.sample([i for i in range(n_nodes)],
                                 int(math.fabs(math.floor(n_nodes*frac_n-1))))

    g = unnormalize_dglfeatures(g)
    key_feat = torch.zeros(n_feats)
    sel_fids = random.sample([i for i in range(n_feats)], int(math.fabs(math.ceil(n_feats*frac_f-1))))
    key_feat[sel_fids] = 1.0
    g = dgl.add_nodes(g, len(sel_nids), {'feat': torch.vstack([key_feat]*len(sel_nids)).cuda()})

    key_nids = [i for i in range(n_nodes, g.num_nodes())]
    g = dgl.add_edges(g, key_nids, sel_nids)

    g.ndata['wm_train_mask'] = torch.BoolTensor([False]*g.num_nodes()).to(g.device)
    g.ndata['wm_train_mask'][sel_nids] = True

    g = dgl.add_self_loop(g)
    g = normalize_dglfeatures(g)

    return g, key_nids, sel_nids, key_feat


def direct_insert_trigger_nodes(g, nids, feat):
    n_nodes = g.num_nodes()
    g = unnormalize_dglfeatures(g)

    g = dgl.add_nodes(g, len(nids), {'feat': torch.vstack([feat]*len(nids)).cuda()})
    key_nids = [i for i in range(n_nodes, g.num_nodes())]
    g = dgl.add_edges(g, key_nids, nids)
    g.ndata['wm_train_mask'] = torch.BoolTensor([False] * g.num_nodes()).to(g.device)
    g.ndata['wm_train_mask'][nids] = True
    g = dgl.add_self_loop(g)
    g = normalize_dglfeatures(g)
    return g

def direct_trig_inject(raw_g, frac_n, frac_f, feat_mode='random_binary'):
    g = deepcopy(raw_g)
    features = g.ndata['feat']
    n_nodes = g.num_nodes()
    n_feats = g.ndata['feat'].shape[-1]
    # sel_nids = random.sample([i for i in range(n_nodes)],
    #                          int(math.fabs(math.floor(n_nodes * frac_n - 1))))
    sel_nids = [i for i in range(int(math.fabs(math.floor(n_nodes * frac_n - 1))))]

    g = unnormalize_dglfeatures(g)
    if feat_mode == 'random_binary':
        # sel_fids = random.sample([i for i in range(n_feats)],
        #                          int(math.fabs(math.ceil(n_feats * frac_f - 1))))
        sel_fids = [i for i in range(int(math.fabs(math.ceil(n_feats * frac_f - 1))))]

        i0, i1 = get_2Dtensor_gridids(sel_nids, sel_fids)
        features[i0, i1] = 1.0

        # a_ = features.cpu().numpy()

        wm_train_mask = torch.BoolTensor([False] * g.num_nodes())
        wm_train_mask[sel_nids] = True
        wm_train_mask = wm_train_mask.cuda()
        g.ndata['wm_train_mask'] = wm_train_mask
        g.ndata['feat'] = features

        g = normalize_dglfeatures(g)
        return g, sel_nids, sel_fids

    else:
        pass


def inject_trig_nodes_epred(g, key_feat, e_pos, e_neg):
    # e_pos, e_neg: dgl form edges, [tensor[u1, ...], tensor[v1, ...]]
    # bi-directed graph

    # pos
    key_feat = binarize_1DTensor(key_feat)
    g = unnormalize_dglfeatures(g)
    n_nodes0 = g.num_nodes()
    g = dgl.add_nodes(g, 2 * e_pos[0].shape[-1],
                      {'feat': torch.vstack([key_feat] * 2 * e_pos[0].shape[-1]).to(g.device)})
    n_nodes1 = g.num_nodes()

    key_nids_pos = [i for i in range(n_nodes0, n_nodes1)]
    g = dgl.add_edges(g, key_nids_pos, e_pos[0].tolist() + e_pos[1].tolist())

    # neg
    g = dgl.add_nodes(g, 2 * e_neg[0].shape[-1],
                      {'feat': torch.vstack([key_feat] * (2 * e_neg[0].shape[-1])).to(g.device)})
    n_nodes2 = g.num_nodes()
    key_nids_neg = [i for i in range(n_nodes1, n_nodes2)]

    g = dgl.add_edges(g, key_nids_neg, e_neg[0].tolist() + e_neg[1].tolist())

    triggered_nodes = e_pos[0].tolist() + e_pos[1].tolist() + e_neg[0].tolist() + e_neg[1].tolist()
    g = dgl.add_self_loop(g)
    g = normalize_dglfeatures(g)

    return g, key_nids_pos, key_nids_neg, triggered_nodes
    # all lists


def binarize_1DTensor(feat):
    nnz_ids = torch.nonzero(feat).squeeze()
    feat[nnz_ids] = 1.0
    return feat

def unnormalize_dglfeatures(g):
    """unnorm the features of g with binary features"""
    g.ndata['feat'] = torch.where(g.ndata['feat'] > 0, 1.0, 0.0).cuda()
    return g

def normalize_dglfeatures(g):
    '''to be tested'''
    normalized_f = normalize_features(g.ndata['feat'].detach().cpu().numpy())
    g.ndata['feat'] = torch.FloatTensor(normalized_f).cuda()
    return g

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(-1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sample_nonedges(g, k):
    # sample #num non-edges from dgl graph g (directed)
    # only efficient for sparse graphs
    src, dst = g.edges()

    # very low probability of sampling positive edges in sparse graphs
    neg_src = src.repeat_interleave(k).cuda()
    neg_dst = torch.randint(0, g.num_nodes(), (len(src) * k,)).cuda()

    return neg_src, neg_dst

def compute_auc(logits, label):
    # [N, 2], [N]
    score_pos = logits[:, 1]
    auc = roc_auc_score(label, score_pos)
    return auc

def get_2Dtensor_gridids(idx0, idx1):
    # For example, idx0 = [0, 1], idx1 = [2, 3]
    # return [0, 0, 1, 1], [2, 2, 3, 3]
    idx_wrapped =  [[i, j] for i in idx0 for j in idx1]
    indices0 = [t[0] for t in idx_wrapped]
    indices1 = [t[1] for t in idx_wrapped]
    return indices0, indices1

def tsne_wm_embeddings(embeds_with_wm, labels, wm_train_mask):
    # embeds: global + local
    n_classes = torch.max(labels).int().item()
    # set the label of watermarked nodes to be unique
    labels = deepcopy(labels)
    labels[wm_train_mask] = n_classes+100

    X = embeds_with_wm.detach().cpu().numpy()
    Y = labels.cpu().numpy()

    tsne_2D = TSNE(n_components=2, init='pca', random_state=22)
    result_2D = tsne_2D.fit_transform(X)

    return result_2D, Y, tsne_2D