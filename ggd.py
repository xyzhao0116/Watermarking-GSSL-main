import torch
import torch.nn as nn
import math
from gcn import GCN
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv
import dgl.function as fn
from wm_utils import *

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, g, features, corrupt=False):
        if corrupt:
            # negative samples
            perm = torch.randperm(g.number_of_nodes())
            features = features[perm]
        if self.gnn_encoder == 'gcn':
            features = self.conv(g, features)

        return features


class GGD(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, proj_layers, gnn_encoder, num_hop):
        super(GGD, self).__init__()
        self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, features, labels, loss_func):
        h_1 = self.encoder(g, features, corrupt=False)  # 正样本头
        h_2 = self.encoder(g, features, corrupt=True)   # 负样本头

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, labels)

        return loss

    def local_embed(self, g, features):
        # local embedding without gradient detach
        h_1 = self.encoder(g, features, corrupt=False)

        return h_1  # h_1: local, h_2: global

    def embed(self, g, features, detach=True):
        h_1 = self.encoder(g, features, corrupt=False)

        feat = h_1.clone().squeeze(0)

        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(h_1.device).unsqueeze(1)
        # Axw + A^{10}x
        for _ in range(10):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'),
                             fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        h_2 = feat.unsqueeze(0)

        if detach:
            return h_1.detach(), h_2.detach()   # h_1: local, h_2: global
        else:
            return h_1, h_2

    def watermark_loss(self, g, g_with_trigger, lambdas=None, loss='mse'):
        # need ndata['wm_train_mask']
        if lambdas is None:
            lambdas = [1.0, 1.0]
        assert "wm_train_mask" in g_with_trigger.ndata.keys(), "node attribute 'wm_train_mask' not in graph"

        wm_train_mask = g_with_trigger.ndata['wm_train_mask']
        wm_train_mask_short = wm_train_mask[:g.num_nodes()]

        h_1 = self.encoder(g, g.ndata['feat'], corrupt=False)
        h_triggered = self.encoder(g_with_trigger,
                                  g_with_trigger.ndata['feat'], corrupt=False)[wm_train_mask]

        h_triggered = torch.nn.functional.normalize(h_triggered, p=2, dim=-1)
        h_1 = torch.nn.functional.normalize(h_1, p=2, dim=-1)

        if loss == 'mse':
            loss_func = nn.MSELoss()    # or cos similarity
        elif loss == 'cos':
            loss_func = nn.CosineSimilarity()

        # l_in
        loss0 = loss_func(torch.vstack([torch.mean(h_triggered, dim=0)]*h_triggered.shape[0]), h_triggered)
        # l_ext
        loss1 = - loss_func(torch.mean(h_1, dim=0, keepdim=True), h_triggered)  #

        if loss == 'cos':
            loss0, loss1 = loss0.mean(), loss1.mean()

        loss = lambdas[0] * loss0 + lambdas[1] * loss1

        print("internal loss {:.4f}, external loss {:.4f}".format(loss0.item(), loss1.item()))
        print("Total watermark loss {:.4f}".format(loss.item()))

        return loss


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)
