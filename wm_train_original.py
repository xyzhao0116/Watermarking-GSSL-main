import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import random
import copy
from ggd import GGD, Classifier, WMExtractor
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import os
from sklearn import preprocessing as sk_prep
from wm_utils import *
from train_downstream import *

def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def load_data_ogb(dataset, args):
    global n_node_feats, n_classes

    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset, root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph

def main(args):
    cuda = True
    free_gpu_id = int(args.gpu)
    torch.cuda.set_device(args.gpu)
    # load and preprocess dataset
    if 'ogbn' not in args.dataset_name:
        args.dataset = args.dataset_name

        data = load_data(args)
        data = data[0]
        features = torch.FloatTensor(data.ndata['feat'])
        labels = torch.LongTensor(data.ndata['label'])
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(data.ndata['train_mask'])
            val_mask = torch.BoolTensor(data.ndata['val_mask'])
            test_mask = torch.BoolTensor(data.ndata['test_mask'])
        else:
            train_mask = torch.ByteTensor(data.ndata['train_mask'])
            val_mask = torch.ByteTensor(data.ndata['val_mask'])
            test_mask = torch.ByteTensor(data.ndata['test_mask'])
        in_feats = features.shape[1]
        n_classes = data.ndata['label'].max().int().item()+1
        n_edges = data.number_of_edges()
        # g = data.graph
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        if args.self_loop:
            data = dgl.add_self_loop(data)
    # else:
        # 再说吧
        # g, labels, train_mask, val_mask, test_mask, evaluator = load_data_ogb(args.dataset_name, args)
        # g = preprocess(g)
        #
        # features = g.ndata['feat']
        # labels = labels.T.squeeze(0)
        #
        # g, labels, train_idx, val_idx, test_idx, features = map(
        #     lambda x: x.to(free_gpu_id), (g, labels, train_mask, val_mask, test_mask, features)
        # )
        #
        # in_feats = g.ndata['feat'].shape[1]
        # n_classes = labels.T.max().item() + 1
        # n_edges = g.num_edges()

    data = data.to(free_gpu_id)
    g_with_trigger, key_nids, sel_nids = select_and_inject_nodes(data, args.frac_n, args.frac_f)

    # create GGD model
    ggd = GGD(in_feats,
              args.n_hidden,
              args.n_layers,
              nn.PReLU(args.n_hidden),
              args.dropout,
              args.proj_layers,
              args.gnn_encoder,
              args.num_hop)

    wm_extractor = WMExtractor(args.n_hidden, args.num_layer_wmextractor, k_wm=args.k_wm, seed=args.seed)


    if cuda:
        ggd.cuda()
        wm_extractor.cuda()

    # ggd_optimizer = torch.optim.AdamW([{'params': ggd.parameters()},
    #                                    {'params': wm_extractor.parameters()}],
    #                                   lr=args.ggd_lr,
    #                                   weight_decay=args.weight_decay)

    ggd_optimizer = torch.optim.AdamW(ggd.parameters(),
                                      lr=args.ggd_lr,
                                      weight_decay=args.weight_decay)

    wm_optimizer = torch.optim.AdamW(wm_extractor.parameters(),
                                     lr=args.wm_lr,
                                     weight_decay=args.wm_weight_decay)

    b_xent = nn.BCEWithLogitsLoss()

    # train deep graph infomax
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

        ggd_optimizer.zero_grad()
        wm_optimizer.zero_grad()

        lbl_1 = torch.ones(1, data.num_nodes())
        lbl_2 = torch.zeros(1, data.num_nodes())    # 用于正负样本节点的伪标签
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        # 看上去是用整个图作为正样本，再复制一份作为负样本，每轮用不同的正负样本
        # 无所谓train_test_val_split，仅在训练下游分类器时使用

        aug_feat = aug_feature_dropout(features, args.drop_feat)

        wm_extract_loss = wm_extractor.wm_loss(ggd.encoder(g_with_trigger, g_with_trigger.ndata['feat'], corrupt=False))


        loss = ggd(data, aug_feat.cuda(), lbl, b_xent) + \
               ggd.constrative_bkd_loss(data, g_with_trigger, [args.lambda0, args.lambda1, args.lambda2]) + \
               args.lambda3 * wm_extract_loss


        loss.backward()
        wm_optimizer.step()
        ggd_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(ggd.state_dict(), './pkl/best_ggd_wm' + tag + '.pkl')
            torch.save(wm_extractor.state_dict(), './pkl/best_wm_extractor' + tag + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            n_edges / np.mean(dur) / 1000))

        counts += 1

    print('Training Completed.')

    # create classifier model
    classifier = Classifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()


    # train classifier

    classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    print('Loading {}th epoch'.format(best_t))

    ggd.load_state_dict(torch.load('./pkl/best_ggd_wm' + tag + '.pkl'))

    # wm_test

    # graph power embedding reinforcement (3.2 Model Inference, global embedding)
    l_embeds, g_embeds = ggd.embed(data, features)   # local, global embeddings

    embeds = (l_embeds + g_embeds).squeeze(0)

    # embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")
    # # L2 row_wise norm
    #
    # embeds = torch.FloatTensor(embeds).cuda()
    embeds = F.normalize(embeds, p=2, dim=-1)

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


    # watermarking performance test
    # 目标节点在被注入攻击后经过encoder，再经过训练好的classifier之后被分类错误的概率
    # 一个节点？训练过的节点？还是很多节点？
    # 先写训练节点吧
    l_embeds_with_wm, g_embeds_with_wm = ggd.embed(g_with_trigger, g_with_trigger.ndata['feat'])

    embeds_with_wm = (l_embeds_with_wm + g_embeds_with_wm).squeeze(0)

    embeds_with_wm = sk_prep.normalize(X=embeds_with_wm.cpu().numpy(), norm="l2")
    # row_wise norm, 各个元素平方和为1

    embeds_with_wm = torch.FloatTensor(embeds_with_wm).cuda()

    wm_train_mask = g_with_trigger.ndata['wm_train_mask']
    labels_with_trigger = g_with_trigger.ndata['label']

    l_embeds_wm = l_embeds_with_wm[wm_train_mask]


    ber = wm_extractor.compute_ber(l_embeds_wm)
    print("BER of Extracted Watermark: {:.4f} of {} total bits".format(ber, args.k_wm))

    attacked_acc = evaluate(classifier, embeds_with_wm, labels_with_trigger, wm_train_mask,
                            plot_lbl_hist=True)

    key_mask = torch.BoolTensor([False]*data.num_nodes())
    key_mask[sel_nids] = True
    # the key nodes in the original non-watermark graph

    raw_acc = evaluate(classifier, embeds, labels, key_mask, plot_lbl_hist=True)
    print("Accuracy of watermarked nodes: {:.4f} vs raw accuracy {:.4f}".format(attacked_acc, raw_acc))


    return best_acc


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='GGD')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--ggd-lr", type=float, default=0.001,
                        help="ggd learning rate")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument("--classifier-lr", type=float, default=0.05,
                        help="classifier learning rate")
    parser.add_argument("--n-ggd-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=6000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--proj_layers", type=int, default=1,
                        help="number of project linear layers")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=500,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--n_trails", type=int, default=1,
                        help="number of trails")
    parser.add_argument("--gnn_encoder", type=str, default='gcn',
                        help="choice of gnn encoder")
    parser.add_argument("--num_hop", type=int, default=10,
                        help="number of k for sgc")
    parser.add_argument('--data_root_dir', type=str, default='default',
                           help="dir_path for saving graph data. Note that this model use DGL loader so do not mix up with the dir_path for the Pyg one. Use 'default' to save datasets at current folder.")
    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Dataset name: cora, citeseer, pubmed, cs, phy')

    # wm
    parser.add_argument('--FTAL', type=bool, default=False,
                        help='whether to fine-tune all the layers when training the downstream classifier')

    parser.add_argument('--num_layer_wmextractor', type=int, default=2)
    parser.add_argument('--k_wm', type=int, default=100)
    parser.add_argument('--seed', type=int, default=121)

    parser.add_argument('--frac_n', type=float, default=0.1)
    parser.add_argument('--frac_f', type=float, default=0.1)

    parser.add_argument('--lambda0', type=float, default=1.0,
                        help='the difference of embeddings between the same nodes after trigger injection')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='the difference of embeddings between the clean and triggered nodes')
    parser.add_argument('--lambda2', type=float, default=30.0,
                        help='the difference of embeddings of all triggered nodes')
    parser.add_argument('--lambda3', type=float, default=1.0,
                        help='watermark extraction')

    parser.add_argument('--wm_lr', type=float, default=0.001)
    parser.add_argument('--wm_weight_decay', type=float, default=0.)


    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    accs = []
    for i in range(args.n_trails):
        accs.append(main(args))
    mean_acc = str(np.array(accs).mean())
    print('mean accuracy:' + mean_acc)

    # file_name = str(args.dataset_name)
    # f = open('result/' + 'result_' + file_name + '.txt', 'a')
    # f.write(str(args) + '\n')
    # f.write(mean_acc + '\n')
