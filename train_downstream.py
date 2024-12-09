import torch
import dgl
import random
import math
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as sk_prep
import time
import torch.nn.functional as F
from wm_utils import *
import networkx as nx
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from sklearn.metrics.cluster import normalized_mutual_info_score

def train_classifier_ncls(args, encoder, classifier, g, features):
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                             lr=args.classifier_lr,
                                             weight_decay=args.weight_decay)

    train_mask, val_mask, test_mask = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
    labels = g.ndata['label']

    l_embeds, g_embeds = encoder.embed(g, features, detach=True) # detached

    if args.add_global:
        embeds = (l_embeds + g_embeds).squeeze(0)
        embeds = F.normalize(embeds, p=2, dim=-1)

    else:
        embeds = l_embeds

    dur = []
    best_acc, best_val_acc = 0, 0
    print('Testing Phase ==== Please Wait.')
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[train_mask], labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_acc = evaluate(classifier, embeds, labels, val_mask)
        if epoch > 1000:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = evaluate(classifier, embeds, labels, test_mask)
                if test_acc > best_acc:
                    best_acc = test_acc
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                     val_acc, n_edges / np.mean(dur) / 1000))
    print("Valid Accuracy {:.4f}".format(best_val_acc))

    # best_acc = evaluate(classifier, embeds, labels, test_mask)
    print("Test Accuracy {:.4f}".format(best_acc))

    return  best_acc


def evaluate_wm_ncls(args, encoder, classifier, g, g_with_trigger):
    l_embeds_with_wm, g_embeds_with_wm = encoder.embed(g_with_trigger, g_with_trigger.ndata['feat'])
    if args.add_global:
        embeds_with_wm = (l_embeds_with_wm + g_embeds_with_wm).squeeze(0)
        embeds_with_wm = F.normalize(embeds_with_wm, p=2, dim=-1)
    else:
        embeds_with_wm = l_embeds_with_wm
        embeds_with_wm = F.normalize(embeds_with_wm, p=2, dim=-1)


    wm_train_mask = g_with_trigger.ndata['wm_train_mask']
    labels_with_trigger = g_with_trigger.ndata['label']

    l_embeds, g_embeds = encoder.embed(g, g.ndata['feat'])
    if args.add_global:
        embeds = (l_embeds + g_embeds).squeeze(0)
        embeds = F.normalize(embeds, p=2, dim=-1)
    else:
        embeds = l_embeds
        embeds = F.normalize(embeds, p=2, dim=-1)

    acc_trig = evaluate(classifier, embeds_with_wm, labels_with_trigger, wm_train_mask)
    cs = evaluate_cs(classifier, embeds_with_wm, wm_train_mask)
    key_mask = wm_train_mask[:g.num_nodes()]

    acc_untrig = evaluate(classifier, embeds, g.ndata['label'], key_mask)
    print("Accuracy of key nodes after triggered: {:.4f} vs before triggered {:.4f}"
          .format(acc_trig, acc_untrig))
    print("concentration score: {:.4f}".format(cs))

    return cs

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)

        return correct.item() * 1.0 / len(labels)

def evaluate_cs(model, features, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        cs = compute_cs(indices)
        return cs

def compute_cs(preds):
    # compute the concentration score of torch.Tensor
    _, counts = preds.unique(return_counts=True)
    max_counts = counts.max()
    cs = max_counts / counts.sum()
    return cs.cpu().item()

def evaluate_auc(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        score_pos = logits[:, 1]
        auc = roc_auc_score(labels.cpu(), score_pos.cpu())
        return auc

def samp_nid_epred_train_test(g, frac_etrain):
    '''
        Sample positive and negative edges from dgl graph g to construct train-test sets
        according to frac_etrain
    '''

    n_edges, pos_edges = g.num_edges(), g.edges()
    indices_edges = [i for i in range(n_edges)]
    random.shuffle(indices_edges)

    num_edges_train_pos = math.ceil(frac_etrain * len(indices_edges))
    num_edges_train_neg = num_edges_train_pos

    num_edges_test_pos = len(indices_edges) - num_edges_train_pos
    num_edges_test_neg = num_edges_test_pos

    eid_edges_train_pos = indices_edges[:num_edges_train_pos]
    eid_edges_test_pos = indices_edges[num_edges_train_pos:]

    neg_src, neg_dst = sample_nonedges(g, 2)   # 2 times of neg edges

    # num of training negative edges is the same as positive ones
    nid_src_train_neg = neg_src[:num_edges_train_neg]
    nid_dst_train_neg = neg_dst[:num_edges_train_neg]

    nid_src_test_neg = neg_src[num_edges_train_neg: num_edges_train_neg + num_edges_test_neg]
    nid_dst_test_neg = neg_dst[num_edges_train_neg: num_edges_train_neg + num_edges_test_neg]

    # train pos+neg
    nid_src_train_pos, nid_dst_train_pos = pos_edges[0][eid_edges_train_pos], pos_edges[1][eid_edges_train_pos]
    # nid_src_train_neg, nid_dst_train_neg = neg_edges[0][eid_edges_train_neg], neg_edges[1][eid_edges_train_neg]

    # test pos+neg
    nid_src_test_pos, nid_dst_test_pos = pos_edges[0][eid_edges_test_pos], pos_edges[1][eid_edges_test_pos]
    # nid_src_test_neg, nid_dst_test_neg = neg_edges[0][eid_edges_test_neg], neg_edges[1][eid_edges_test_neg]

    nid_src_train = torch.cat([nid_src_train_pos, nid_src_train_neg])
    nid_dst_train = torch.cat([nid_dst_train_pos, nid_dst_train_neg])
    nid_src_test = torch.cat([nid_src_test_pos, nid_src_test_neg])
    nid_dst_test = torch.cat([nid_dst_test_pos, nid_dst_test_neg])

    lbl_train = torch.cat([torch.ones(nid_src_train_pos.shape[-1]), torch.zeros(nid_src_train_neg.shape[-1])])
    lbl_test = torch.cat([torch.ones(nid_src_test_pos.shape[-1]), torch.zeros(nid_src_test_neg.shape[-1])])

    lbl_train = lbl_train.long().cuda()
    lbl_test = lbl_test.long().cuda()

    return nid_src_train.tolist(), nid_dst_train.tolist(), nid_src_test.tolist(), nid_dst_test.tolist(), \
           lbl_train, lbl_test


def train_classifier_epred(args, edges_train, edges_test, encoder, classifier, g, features, lbls):
    l_embeds, g_embeds = encoder.embed(g, features)  # not detached
    if args.add_global:
        embeds = (l_embeds + g_embeds).squeeze(0)
        embeds = F.normalize(embeds, p=2, dim=-1)
    else:
        embeds = l_embeds
        embeds = F.normalize(embeds, p=2, dim=-1)

    lbl_train, lbl_test = lbls[0].long().cuda(), lbls[1].long().cuda()

    src_embeds_train, dst_embeds_train = embeds[edges_train[0]], embeds[edges_train[1]]
    src_embeds_test, dst_embeds_test = embeds[edges_test[0]], embeds[edges_test[1]]

    pos_src = src_embeds_train[:int(src_embeds_train.shape[0] / 2)]
    pos_dst = dst_embeds_train[:int(dst_embeds_train.shape[0] / 2)]
    neg_src = src_embeds_train[int(src_embeds_train.shape[0] / 2):]
    neg_dst = src_embeds_train[int(dst_embeds_train.shape[0] / 2):]

    func = torch.nn.MSELoss()
    pos_sim = func(pos_src, pos_dst)
    neg_sim = func(neg_src, neg_dst)
    print(pos_sim, neg_sim)

    embeds_train = torch.cat([src_embeds_train, dst_embeds_train], dim=1)
    embeds_test = torch.cat([src_embeds_test, dst_embeds_test], dim=1)

    classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                             lr=args.classifier_lr,
                                             weight_decay=args.weight_decay)

    dur = []
    print('Testing Phase ==== Please Wait.')
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds_train)
        loss = F.nll_loss(preds, lbl_train)
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

    train_mask = torch.BoolTensor([True]*lbl_train.shape[0]).cuda()
    train_acc = evaluate(classifier, embeds_train, lbl_train, train_mask)
    auc_train = evaluate_auc(classifier, embeds_train, lbl_train, train_mask)
    print("Training acc {:.4f}, auc {:.4f}".format(train_acc, auc_train))

    test_mask = torch.BoolTensor([True]*lbl_test.shape[0]).cuda()
    best_acc = evaluate(classifier, embeds_test, lbl_test, test_mask)
    auc = evaluate_auc(classifier, embeds_test, lbl_test, test_mask)
    print("Test auc: {:.4f}".format(auc))
    print("Test Accuracy {:.4f}".format(best_acc))

    return best_acc

def evaluate_wm_epred(args, encoder, classifier, g, trig_ebd):
    # randomly sample some edges and non-edges
    edges = g.edges()
    num_sample = math.ceil(args.frac_e_validate * g.num_edges())
    eid_val_pos = random.sample([i for i in range(g.num_nodes())], num_sample)
    src_val_pos, dst_val_pos = edges[0][eid_val_pos], edges[1][eid_val_pos] # torch.Tensor

    src_neg, dst_neg = sample_nonedges(g, 5)
    eid_val_neg = random.sample([i for i in range(g.num_nodes())], num_sample)
    src_val_neg, dst_val_neg = src_neg[eid_val_neg], dst_neg[eid_val_neg]

    g_trigger, key_nids_pos, key_nids_neg, triggered_nodes = \
        inject_trig_nodes_epred(g, trig_ebd, [src_val_pos, dst_val_pos], [src_val_neg, dst_val_neg])

    l_embeds_t, g_embeds_t = encoder.embed(g_trigger, g_trigger.ndata['feat'])
    embeds_t = (l_embeds_t + g_embeds_t).squeeze(0)
    embeds_t = F.normalize(embeds_t, p=2, dim=-1)

    ## accuracy of attacked edges
    embeds_pos = torch.cat([embeds_t[src_val_pos], embeds_t[dst_val_pos]], dim=1)
    embeds_neg = torch.cat([embeds_t[src_val_neg], embeds_t[dst_val_neg]], dim=1)

    embeds_key = torch.vstack([embeds_pos, embeds_neg])
    lbl = torch.cat([torch.ones(embeds_pos.shape[0]), torch.zeros(embeds_neg.shape[0])]).cuda()

    acc_trig = evaluate(classifier, embeds_key, lbl, torch.BoolTensor([True] * lbl.shape[0]))
    auc_trig = evaluate_auc(classifier, embeds_key, lbl, torch.BoolTensor([True] * lbl.shape[0]))
    cs = evaluate_cs(classifier, embeds_key, torch.BoolTensor([True] * lbl.shape[0]))
    print("Edge Prediction Accuracy of Attacked edges: {:.4f}, AUC: {:.4f}".format(acc_trig, auc_trig))
    print("concentration score: {:.4f}".format(cs))

    # accuracy of the raw un-attacked edges
    l_embeds_r, g_embeds_r = encoder.embed(g, g.ndata['feat'])  # not detached
    embeds_r = (l_embeds_r + g_embeds_r).squeeze(0)
    embeds_r = F.normalize(embeds_r, p=2, dim=-1)
    embeds_pos_r = torch.cat([embeds_r[src_val_pos], embeds_r[dst_val_pos]], dim=1)
    embeds_neg_r = torch.cat([embeds_r[src_val_neg], embeds_r[dst_val_neg]], dim=1)
    embeds_key_r = torch.vstack([embeds_pos_r, embeds_neg_r])
    acc_trig_raw = evaluate(classifier, embeds_key_r, lbl, torch.BoolTensor([True] * lbl.shape[0]))
    auc_trig_raw = evaluate_auc(classifier, embeds_key_r, lbl, torch.BoolTensor([True] * lbl.shape[0]))

    print("Edge Prediction Accuracy of the Raw Un-attacked selected edges: {:.4f}, AUC: {:.4f}".format(
        acc_trig_raw, auc_trig_raw))

    return cs

