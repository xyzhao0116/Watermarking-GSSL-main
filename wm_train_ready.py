import os
import argparse
from dgl.data import register_data_args, load_data
from ggd import GGD, Classifier
from train_downstream import *
from optim import *
from sklearn.cluster import KMeans


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
    args.dataset = args.dataset_name
    data = load_dgl_cite_dataset(args.dataset_name, dir='/xyzhao/datasets', local=True)

    data = data[0]
    features = torch.FloatTensor(data.ndata['feat'])

    in_feats = features.shape[1]
    n_classes = data.ndata['label'].max().int().item()+1
    # g = data.graph
    features = features.cuda()

    if args.self_loop:
        data = dgl.add_self_loop(data)

    data = data.to(free_gpu_id)
    g_with_trigger, key_nids, sel_nids, key_feat = select_and_inject_nodes(data, args.frac_n, args.frac_f)

    # create GGD model
    ggd = GGD(in_feats,
              args.n_hidden,
              args.n_layers,
              nn.PReLU(args.n_hidden),
              args.dropout,
              args.proj_layers,
              args.gnn_encoder,
              args.num_hop)

    if cuda:
        ggd.cuda()

    watermark_cotraining(args, ggd, data, g_with_trigger)

    # downstream verification
    if args.task == 'ncls':
        classifier = Classifier(args.n_hidden, n_classes).cuda()
        train_classifier_ncls(args, ggd, classifier, data, features)
        # g_with_trigger = direct_insert_trigger_nodes(data, sel_nids, key_feat)
        cs = evaluate_wm_ncls(args, ggd, classifier, data, g_with_trigger)
        return cs

    elif args.task == 'epred':
        classifier = Classifier(2*args.n_hidden, 2).cuda()

        nid_src_train, nid_dst_train, nid_src_test, nid_dst_test, lbl_train, lbl_test \
            = samp_nid_epred_train_test(data, args.frac_etrain)

        edges_train = [nid_src_train, nid_dst_train]    # list of lists
        edges_test = [nid_src_test, nid_dst_test]       # list of lists
        lbls = [lbl_train, lbl_test]

        train_classifier_epred(args, edges_train, edges_test, ggd, classifier, data, data.ndata['feat'], lbls)
        cs = evaluate_wm_epred(args, ggd, classifier, data, g_with_trigger.ndata['feat'][-1])

        return cs


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Watermarking-GSSL-GGD')
    register_data_args(parser)

    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Dataset name: cora, citeseer')
    parser.add_argument('--task', type=str, default='ncls', choices=['ncls', 'epred'],
                        help='only support node classification and edge prediction for now')
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--ggd-lr", type=float, default=0.001,
                        help="ggd learning rate")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument("--classifier-lr", type=float, default=0.001,
                        help="classifier learning rate")
    parser.add_argument("--n-ggd-epochs", type=int, default=1000,
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

    # wm
    parser.add_argument('--add_global', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=225)

    parser.add_argument('--frac_n', type=float, default=0.1)
    parser.add_argument('--frac_f', type=float, default=0.2)

    # edge prediction
    parser.add_argument('--frac_etrain', type=float, default=0.9)
    parser.add_argument('--frac_e_validate', type=float, default=0.01)

    # optimization
    parser.add_argument('--loss_func', type=str, default='mse',
                        help='[cos, mse]')
    parser.add_argument("--n-classifier-epochs", type=int, default=6000,   # 6000
                        help="number of training epochs")
    parser.add_argument('--lambda0', type=float, default=10.0,
                        help='internal loss to enclose the trigger embeddings')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='external loss to distinguish between normal embeddings and trigger embeddings')

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    css = []
    for i in range(args.n_trails):
        css.append(main(args))
    mean_cs = np.array(css).mean()
    print('mean concentration score: {:.4f}'.format(mean_cs))

