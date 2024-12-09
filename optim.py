import torch
import dgl
import networkx as nx
import random
import math
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import time
from wm_utils import *
import torch.nn as nn


def normal_pretraining(args, ggd, g):
    opt_ggd = torch.optim.AdamW(ggd.parameters(),
                                lr=args.ggd_lr,
                                weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    dur = []

    for epoch in range(args.n_ggd_epochs):
        ggd.train()
        if epoch >= 3:
            t0 = time.time()

        opt_ggd.zero_grad()

        lbl_1 = torch.ones(1, g.num_nodes())
        lbl_2 = torch.zeros(1, g.num_nodes())
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        aug_feat = aug_feature_dropout(g.ndata['feat'], args.drop_feat)

        loss = ggd(g, aug_feat.cuda(), lbl, b_xent)
        print("loss: {:4f}".format(loss.item()))

        loss.backward()
        opt_ggd.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            g.num_edges() / np.mean(dur) / 1000))

        counts += 1

    print('Normal Pre-training Completed.')
    return ggd


def watermark_cotraining(args, ggd, g, g_wt):
    # watermarking via pretraining

    opt_ggd = torch.optim.AdamW(ggd.parameters(),
                                      lr=args.ggd_lr,
                                      weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    dur = []

    tag = str(int(np.random.random() * 10000000000))
    for epoch in range(args.n_ggd_epochs):
        ggd.train()
        if epoch >= 3:
            t0 = time.time()

        opt_ggd.zero_grad()

        lbl_1 = torch.ones(1, g.num_nodes())
        lbl_2 = torch.zeros(1, g.num_nodes())
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        aug_feat = aug_feature_dropout(g.ndata['feat'], args.drop_feat)

        loss = ggd(g, aug_feat.cuda(), lbl, b_xent) + \
               ggd.watermark_loss(g, g_wt, [args.lambda0, args.lambda1], args.loss_func)

        loss.backward()
        opt_ggd.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(ggd.state_dict(), './pkl/best_ggd_wm' + tag + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            g.num_edges() / np.mean(dur) / 1000))

        counts += 1

    print('Co-Training Completed.')
    return



