import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
from util import *
# from util import load_data, separate_data, backdoor_random, backdoor_degree, backdoor_graph_generation
from models.graphcnn import GraphCNN
import pickle
import copy

criterion = nn.CrossEntropyLoss()

def matrix_sample(n, k):
    # n*n symmetrix matrix sample k units
    sample_list = random.sample(range(1, int(n * (n - 1) / 2 + 1)), k)
    A = np.zeros((n, n))
    for m in sample_list:
        i = 0
        l = n - 1
        while (m > l):
            m -= l
            i += 1
            l -= 1
        j = i + m
        A[i][j] = A[j][i] = 1
    return A

def adversarial_train(args, batch_graph, tag2index):
    noise_sample_list = []
    for g in batch_graph:
        n = len(g.g)
        sample = copy.deepcopy(g)

        k = int(args.randomly_preserve * n * (n - 1) / 2)
        mask = matrix_sample(n, k)
        adj_matrix = nx.to_numpy_array(sample.g) * mask
        sample.g = nx.to_networkx_graph(adj_matrix)
        edges = list(sample.g.edges)
        edges.extend([[i, j] for j, i in edges])
        sample.node_tags = list(dict(sample.g.degree).values())
        sample.node_features = torch.zeros(len(sample.node_tags), len(tag2index))
        sample.node_features[range(len(sample.node_tags)), [tag2index[tag] for tag in sample.node_tags]] = 1

        if len(edges) == 0:
            sample.edge_mat = torch.LongTensor([[], []])
        else:
            sample.edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        noise_sample_list.append(sample)

    return noise_sample_list

def train(args, model, device, train_graphs, optimizer, epoch, tag2index):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        if args.adversarial_training:
            batch_graph = adversarial_train(args, batch_graph, tag2index)
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(args, model, device, test_graphs, tag2index):
    model.eval()
    if args.adversarial_training:
        test_graphs = adversarial_train(args, test_graphs, tag2index)

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # print(labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy test: %f" % acc_test)

    return acc_test


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--backdoor', action='store_true', default=True,
                        help='Backdoor GNN')
    parser.add_argument('--graphtype', type=str, default='ER',
                        help='type of graph generation')
    parser.add_argument('--prob', type=float, default=0.5,
                        help='probability for edge creation/rewiring each edge')
    parser.add_argument('--K', type=int, default=4,
                        help='Each node is connected to k nearest neighbors in ring topology')
    parser.add_argument('--frac', type=float, default=0.01,
                        help='fraction of training graphs are backdoored')
    parser.add_argument('--triggersize', type=int, default=3,
                        help='number of nodes in a clique (trigger size)')
    parser.add_argument('--target', type=int, default=0,
                        help='targe class (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=False,
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="output",
                        help='output file')
    parser.add_argument('--filenamebd', type=str, default="output_bd",
                        help='output backdoor file')
    parser.add_argument('--adversarial_training', action="store_true", default=False,
                        help='whether to use adversarial training in the training phase')
    parser.add_argument('--randomly_preserve', type=float, default=0.1,
                        help='when adversarial_training, randomly preserve certain fraction of entries')
    args = parser.parse_args()

    if args.adversarial_training:
        print('adversarial_training')
    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes, tag2index = load_data(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs, test_idx = separate_data(graphs, args.seed, args.fold_idx)

    print('#train_graphs:', len(train_graphs), '#test_graphs:', len(test_graphs))

    test_cleangraph_backdoor_labels = [graph for graph in test_graphs if graph.label != args.target]
    print('#test clean graphs:', len(test_cleangraph_backdoor_labels))

    print('input dim:', train_graphs[0].node_features.shape[1])


    if args.backdoor:

        train_backdoor, test_backdoor_labels = backdoor_graph_generation_random(args.dataset, args.degree_as_tag,
                                                                                args.frac, args.triggersize, args.seed,
                                                                                args.fold_idx, args.target,
                                                                                args.graphtype, args.prob, args.K, tag2index)
        # train_backdoor, test_backdoor_labels = backdoor_graph_generation_degree(args.dataset, args.degree_as_tag, args.frac, args.triggersize, args.seed, args.fold_idx, args.target, args.graphtype, args.prob, args.K)
        for g in test_backdoor_labels:
            g.label = args.target

        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_backdoor[0].node_features.shape[1],
                         args.hidden_dim, \
                         num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                         args.neighbor_pooling_type, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        with open(args.filenamebd, 'w+') as f:
            f.write("acc_train acc_clean acc_backdoor\n")

            for epoch in range(1, args.epochs + 1):
                scheduler.step()
                avg_loss = train(args, model, device, train_backdoor, optimizer, epoch, tag2index)
                if epoch%5 ==0:
                    #acc_train=test(args, model, device, train_backdoor, tag2index)
                    acc_test_clean = test(args, model, device, test_graphs, tag2index)
                    acc_test_backdoor = test(args, model, device, test_backdoor_labels, tag2index)
                    f.write("%f %f\n" % (acc_test_clean, acc_test_backdoor))
                    #f.write("%f\n" % acc_test_clean)
                    f.flush()

        if args.adversarial_training:
            f = open('saved_model/' + str(args.graphtype) + '_' + str(args.dataset) + '_' + str(
                args.frac) + '_triggersize_' + str(args.triggersize) + '_entries_preserved_' + str(
                args.randomly_preserve) + '_adversarial_training', 'wb')
        else:
            f = open('saved_model/' + str(args.graphtype) + '_' + str(args.dataset) + '_' + str(
                args.frac) + '_triggersize_' + str(args.triggersize), 'wb')

        pickle.dump(model, f)
        f.close()


if __name__ == '__main__':
    main()