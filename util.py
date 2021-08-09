import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_callgraph():
    import pickle

    dataset = "./dataset/MALWARE/dataset/"
    list_of_malware_graph = pickle.load(open(dataset + 'malware_graphs.p', "rb"))
    list_of_goodware_graph = pickle.load(open(dataset + 'goodware_graphs.p', "rb"))

    print(list_of_malware_graph[0])
    print(list_of_malware_graph[1])


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            if n<400:
                g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)


    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
            # using degree of nodes as tags

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    max_degree=max(tagset)
    tag2index = {i: i for i in range(max_degree+1)}
    #tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        #g.node_features = torch.zeros(len(g.node_tags), 1)
        #g.node_features[range(len(g.node_tags)), [0]] = 1
        g.node_features = torch.zeros(len(g.node_tags), len(tag2index))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tag2index))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict), tag2index

#def most_connected(g,number):
   # select_nodes=[]
   # while(len(select_nodes)<number):
        
   # return select_nodes

def backdoor_graph_generation_random(dataset, degree_as_tag, frac, num_backdoor_nodes, seed, fold_idx, target_label,
                                     graph_type, prob, K, tag2index):
    ## erdos_renyi
    if graph_type == 'ER':
        print(np.log(num_backdoor_nodes) / num_backdoor_nodes)
        #assert prob > np.log(num_backdoor_nodes) / num_backdoor_nodes
        G_gen = nx.erdos_renyi_graph(num_backdoor_nodes, prob)
        nx.write_edgelist(G_gen, 'subgraph_gen/ER_' + str(dataset) + '_triggersize_' + str(
            num_backdoor_nodes) + '_prob_' + str(prob) + '.edgelist')
        test_graph_file = open(
            'test_graphs/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_file = open(
            'backdoor_graphs/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphnodes', 'w')
        # G_gen = nx.read_edgelist('subgraph_gen/ER_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_prob_'+str(prob)+'.edgelist')

    ## small_world: Watts-Strogatz small-world graph
    # K: Each node is connected to k nearest neighbors in ring topology
    # p: The probability of rewiring each edge
    if graph_type == 'SW':
        assert num_backdoor_nodes > K
        G_gen = nx.watts_strogatz_graph(num_backdoor_nodes, K, prob, seed=None)
        nx.write_edgelist(G_gen,
                          'subgraph_gen/SW_' + str(dataset) + '_triggersize_' + str(num_backdoor_nodes) + '_NN_' + str(
                              K) + '_prob_' + str(prob) + '.edgelist')
        # G_gen = nx.read_edgelist('subgraph_gen/SW_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_NN_'+str(K)+'_prob_'+str(prob)+'.edgelist')
        test_graph_file = open(
            'test_graphs/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_file = open(
            'backdoor_graphs/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphnodes', 'w')

    ## preferential_attachment: scale-free power-law Barabási–Albert preferential attachment model.
    # K: Number of edges to attach from a new node to existing nodes
    if graph_type == 'PA':
        G_gen = nx.barabasi_albert_graph(num_backdoor_nodes, K, seed=None)
        nx.write_edgelist(G_gen, 'subgraph_gen/PA_' + str(dataset) + 'frac_' + str(frac) + '_triggersize_' + str(
            num_backdoor_nodes) + '_edgeattach_' + str(K) + '.edgelist')
        test_graph_file = open(
            'test_graphs/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        # G_gen = nx.read_edgelist('subgraph_gen/PA_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_edgeattach_'+str(K)+'.edgelist')
        train_graph_file = open(
            'backdoor_graphs/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphnodes', 'w')

    # print(G_gen.nodes)
    # print(G_gen.edges)

    graphs, num_classes, tag2index = load_data(dataset, degree_as_tag)
    train_graphs, test_graphs, test_idx = separate_data(graphs, seed, fold_idx)
    print('#train:', len(train_graphs), '#class:', num_classes)

    num_backdoor_train_graphs = int(frac * len(train_graphs))
    print('#backdoor graphs:', num_backdoor_train_graphs)

    # Backdoor: target class: target_label
    # label 1,2,... -> target_label
    train_graphs_target_label_indexes = []
    train_backdoor_graphs_indexes = []

    for graph_idx in range(len(train_graphs)):
        if train_graphs[graph_idx].label == target_label:
            train_graphs_target_label_indexes.append(graph_idx)
        else:
            train_backdoor_graphs_indexes.append(graph_idx)
    print('#train target label:', len(train_graphs_target_label_indexes), '#train backdoor labels:',
          len(train_backdoor_graphs_indexes))

    rand_backdoor_graph_idx = random.sample(train_backdoor_graphs_indexes,
                                            k=num_backdoor_train_graphs)  # without replacement

    train_graph_file.write(" ".join(str(idx) for idx in rand_backdoor_graph_idx))
    train_graph_file.close()

    for idx in rand_backdoor_graph_idx:
        # print(train_graphs[idx].edge_mat)
        num_nodes = torch.max(train_graphs[idx].edge_mat).numpy() + 1
        # print('#nodes:', num_nodes)
        if num_backdoor_nodes >= num_nodes:
            # rand_select_nodes = [node for node in range(num_nodes)]
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)
        # print('select nodes:', rand_select_nodes)

        train_graph_nodefile.write(" ".join(str(idx) for idx in rand_select_nodes))
        train_graph_nodefile.write("\n")

        edges = train_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()
        # print('raw edges:', edges)
        # print('#raw edges:', len(edges))

        ### Remove existing edges
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i, j) in train_graphs[idx].g.edges():
                    train_graphs[idx].g.remove_edge(i, j)
        # print('after remove:', len(edges))

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
            train_graphs[idx].g.add_edge(e[0], e[1])
        # print('after add:', len(edges))
        # print('new edges:', edges)

        train_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        train_graphs[idx].label = target_label
        train_graphs[idx].node_tags = list(dict(train_graphs[idx].g.degree).values())
        train_graphs[idx].node_features = torch.zeros(len(train_graphs[idx].node_tags), len(tag2index))
        train_graphs[idx].node_features[range(len(train_graphs[idx].node_tags)), [tag2index[tag] for tag in train_graphs[idx].node_tags]] = 1
        # print(train_graphs[idx].edge_mat)

    train_graph_nodefile.close()

    # train_labels = torch.LongTensor([graph.label for graph in train_graphs])
    # print(train_labels)

    test_graphs_targetlabel_indexes = []
    test_backdoor_graphs_indexes = []
    for graph_idx in range(len(test_graphs)):
        if test_graphs[graph_idx].label != target_label:
            test_backdoor_graphs_indexes.append(graph_idx)
        else:
            test_graphs_targetlabel_indexes.append(graph_idx)
    print('#test target label:', len(test_graphs_targetlabel_indexes), '#test backdoor labels:',
          len(test_backdoor_graphs_indexes))

    test_graph_file.write(" ".join(str(idx) for idx in test_idx))
    test_graph_file.close()

    for idx in test_backdoor_graphs_indexes:
        num_nodes = torch.max(test_graphs[idx].edge_mat).numpy() + 1
        # print('#nodes:', num_nodes)
        if num_backdoor_nodes >= num_nodes:
            # rand_select_nodes = [node for node in range(num_nodes)]
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)
        # print('select nodes:', rand_select_nodes)
        edges = test_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()
        # print('raw edges:', edges)
        # print('#raw edges:', len(edges))

        ### Remove existing edges
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i,j) in test_graphs[idx].g.edges():
                    test_graphs[idx].g.remove_edge(i, j)
        # print('after remove:', len(edges))

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
            test_graphs[idx].g.add_edge(e[0], e[1])
        # print('after add:', len(edges))
        # print('new edges:', edges)

        test_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        test_graphs[idx].node_tags = list(dict(test_graphs[idx].g.degree).values())
        test_graphs[idx].node_features = torch.zeros(len(test_graphs[idx].node_tags), len(tag2index))
        test_graphs[idx].node_features[range(len(test_graphs[idx].node_tags)), [tag2index[tag] for tag in test_graphs[idx].node_tags]] = 1

    test_backdoor_graphs = [graph for graph in test_graphs if graph.label != target_label]

    return train_graphs, test_backdoor_graphs


def backdoor_graph_generation_degree(dataset, degree_as_tag, frac, num_backdoor_nodes, seed, fold_idx, target_label,
                                     graph_type, prob, K,tag2index):
    ## erdos_renyi
    if graph_type == 'ER':
        print(np.log(num_backdoor_nodes) / num_backdoor_nodes)
        assert prob > np.log(num_backdoor_nodes) / num_backdoor_nodes
        G_gen = nx.erdos_renyi_graph(num_backdoor_nodes, prob)
        nx.write_edgelist(G_gen, 'subgraph_gen_deg/ER_' + str(dataset) + '_triggersize_' + str(
            num_backdoor_nodes) + '_prob_' + str(prob) + '.edgelist')

        train_graph_file = open(
            'backdoor_graphs_deg/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs_deg/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphnodes', 'w')
        # G_gen = nx.read_edgelist('subgraph_gen_deg/ER_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_prob_'+str(prob)+'.edgelist')

        #rand_backdoor_graph_idx = np.loadtxt(
            #'backdoor_graphs/ER/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
              #  num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', dtype=int)


    ## small_world: Watts-Strogatz small-world graph
    # K: Each node is connected to k nearest neighbors in ring topology
    # p: The probability of rewiring each edge
    if graph_type == 'SW':
        assert num_backdoor_nodes > K
        G_gen = nx.watts_strogatz_graph(num_backdoor_nodes, K, prob, seed=None)
        nx.write_edgelist(G_gen, 'subgraph_gen_deg/SW_' + str(dataset) + '_triggersize_' + str(
            num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.edgelist')
        # G_gen = nx.read_edgelist('subgraph_gen_deg/SW_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_NN_'+str(K)+'_prob_'+str(prob)+'.edgelist')

        train_graph_file = open(
            'backdoor_graphs_deg/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs_deg/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphnodes', 'w')

        rand_backdoor_graph_idx = np.loadtxt(
            'backdoor_graphs/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphs', dtype=int)
        print('#backdoor graphs:', len(rand_backdoor_graph_idx))

    ## preferential_attachment: scale-free power-law Barabási–Albert preferential attachment model.
    # K: Number of edges to attach from a new node to existing nodes
    if graph_type == 'PA':
        G_gen = nx.barabasi_albert_graph(num_backdoor_nodes, K, seed=None)
        nx.write_edgelist(G_gen, 'subgraph_gen_deg/PA_' + str(dataset) + 'frac_' + str(frac) + '_triggersize_' + str(
            num_backdoor_nodes) + '_edgeattach_' + str(K) + '.edgelist')

        # G_gen = nx.read_edgelist('subgraph_gen_deg/PA_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_edgeattach_'+str(K)+'.edgelist')
        train_graph_file = open(
            'backdoor_graphs_deg/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs_deg/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphnodes', 'w')

        rand_backdoor_graph_idx = np.loadtxt(
            'backdoor_graphs_deg/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphs', dtype=int)
        print('#backdoor graphs:', len(rand_backdoor_graph_idx))

    # print(G_gen.nodes)
    # print(G_gen.edges)

    graphs, num_classes, tag2index = load_data(dataset, degree_as_tag)
    train_graphs, test_graphs, test_idx = separate_data(graphs, seed, fold_idx)
    print('#train:', len(train_graphs), '#class:', num_classes)

    num_backdoor_train_graphs = int(frac * len(train_graphs))
    print('#backdoor graphs:', num_backdoor_train_graphs)

    # # Backdoor: target class: target_label
    # # label 1,2,... -> target_label
    train_graphs_target_label_indexes = []
    train_backdoor_graphs_indexes = []

    for graph_idx in range(len(train_graphs)):
        if train_graphs[graph_idx].label == target_label:
            train_graphs_target_label_indexes.append(graph_idx)
        else:
            train_backdoor_graphs_indexes.append(graph_idx)

    print('#train target label:', len(train_graphs_target_label_indexes), '#train backdoor labels:', len(train_backdoor_graphs_indexes))

    rand_backdoor_graph_idx = random.sample(train_backdoor_graphs_indexes, k=num_backdoor_train_graphs) # without replacement

    for idx in rand_backdoor_graph_idx:
        degree_list = []
        # print(train_graphs[idx].edge_mat)
        num_nodes = torch.max(train_graphs[idx].edge_mat).numpy() + 1

        for node in range(num_nodes):
            degree_list.append((node, len(train_graphs[idx].neighbors[node])))
        # print('degree list:', degree_list)
        select_nodes = sorted(degree_list, key=lambda k: k[1], reverse=True)
        rand_select_nodes= [node[0] for node in select_nodes[:num_backdoor_nodes]]
        #assert len(rand_select_nodes)==num_backdoor_nodes
# most connected subgraph
        #select_nodes=most

        edges = train_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()
        ### Remove existing edges between selected nodes
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
        # print('after remove:', len(edges))

        for e in G_gen.edges:
            if e[0]<num_nodes and e[1]<num_nodes:
            # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
        # print('after add:', len(edges))
        # print('new edges:', edges)

        train_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        train_graphs[idx].label = target_label
        train_graphs[idx].node_tags = list(dict(train_graphs[idx].g.degree).values())
        train_graphs[idx].node_features = torch.zeros(len(train_graphs[idx].node_tags), len(tag2index))
        train_graphs[idx].node_features[range(len(train_graphs[idx].node_tags)), [tag2index[tag] for tag in train_graphs[idx].node_tags]] = 1

    # train_labels = torch.LongTensor([graph.label for graph in train_graphs])
    # print(train_labels)

    test_graphs_targetlabel_indexes = []
    test_backdoor_graphs_indexes = []
    for graph_idx in range(len(test_graphs)):
        if test_graphs[graph_idx].label != target_label:
            test_backdoor_graphs_indexes.append(graph_idx)
        else:
            test_graphs_targetlabel_indexes.append(graph_idx)
    print('#test target label:', len(test_graphs_targetlabel_indexes), '#test backdoor labels:',
          len(test_backdoor_graphs_indexes))

    for idx in test_backdoor_graphs_indexes:
        degree_list = []
        num_nodes = torch.max(test_graphs[idx].edge_mat).numpy() + 1

        for node in range(num_nodes):
            degree_list.append((node,len(test_graphs[idx].neighbors[node])))
        # print('degree list:', degree_list)
        select_nodes = sorted(degree_list, key=lambda k: k[1], reverse=True)
        # print('select nodes:', select_nodes)
        rand_select_nodes = [node[0] for node in select_nodes[:num_backdoor_nodes]]
        # print('select nodes:', rand_select_nodes)
        edges = test_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()

        ### Remove existing edges
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
        # print('after remove:', len(edges))

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            if e[0]<num_nodes and e[1]<num_nodes:
                # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
        # print('after add:', len(edges))
        # print('new edges:', edges)

        test_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        test_graphs[idx].node_tags = list(dict(test_graphs[idx].g.degree).values())
        test_graphs[idx].node_features = torch.zeros(len(test_graphs[idx].node_tags), len(tag2index))
        test_graphs[idx].node_features[range(len(test_graphs[idx].node_tags)), [tag2index[tag] for tag in test_graphs[idx].node_tags]] = 1

    test_backdoor_graphs = [graph for graph in test_graphs if graph.label != target_label]

    return train_graphs, test_backdoor_graphs
def backdoor_graph_generation_connect(dataset, degree_as_tag, frac, num_backdoor_nodes, seed, fold_idx, target_label,
                                     graph_type, prob, K,tag2index):
    ## erdos_renyi
    if graph_type == 'ER':
        print(np.log(num_backdoor_nodes) / num_backdoor_nodes)
        assert prob > np.log(num_backdoor_nodes) / num_backdoor_nodes
        G_gen = nx.erdos_renyi_graph(num_backdoor_nodes, prob)
        nx.write_edgelist(G_gen, 'subgraph_gen_deg/ER_' + str(dataset) + '_triggersize_' + str(
            num_backdoor_nodes) + '_prob_' + str(prob) + '.edgelist')

        train_graph_file = open(
            'backdoor_graphs_deg/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs_deg/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphnodes', 'w')
        # G_gen = nx.read_edgelist('subgraph_gen_deg/ER_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_prob_'+str(prob)+'.edgelist')

        #rand_backdoor_graph_idx = np.loadtxt(
            #'backdoor_graphs/ER/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
             #   num_backdoor_nodes) + '_prob_' + str(prob) + '.backdoor_graphs', dtype=int)


    ## small_world: Watts-Strogatz small-world graph
    # K: Each node is connected to k nearest neighbors in ring topology
    # p: The probability of rewiring each edge
    if graph_type == 'SW':
        assert num_backdoor_nodes > K
        G_gen = nx.watts_strogatz_graph(num_backdoor_nodes, K, prob, seed=None)
        nx.write_edgelist(G_gen, 'subgraph_gen_deg/SW_' + str(dataset) + '_triggersize_' + str(
            num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.edgelist')
        # G_gen = nx.read_edgelist('subgraph_gen_deg/SW_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_NN_'+str(K)+'_prob_'+str(prob)+'.edgelist')

        train_graph_file = open(
            'backdoor_graphs_deg/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs_deg/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphnodes', 'w')

        rand_backdoor_graph_idx = np.loadtxt(
            'backdoor_graphs/SW/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_NN_' + str(K) + '_prob_' + str(prob) + '.backdoor_graphs', dtype=int)
        print('#backdoor graphs:', len(rand_backdoor_graph_idx))

    ## preferential_attachment: scale-free power-law Barabási–Albert preferential attachment model.
    # K: Number of edges to attach from a new node to existing nodes
    if graph_type == 'PA':
        G_gen = nx.barabasi_albert_graph(num_backdoor_nodes, K, seed=None)
        nx.write_edgelist(G_gen, 'subgraph_gen_deg/PA_' + str(dataset) + 'frac_' + str(frac) + '_triggersize_' + str(
            num_backdoor_nodes) + '_edgeattach_' + str(K) + '.edgelist')

        # G_gen = nx.read_edgelist('subgraph_gen_deg/PA_'+str(dataset)+'_triggersize_'+str(num_backdoor_nodes)+'_edgeattach_'+str(K)+'.edgelist')
        train_graph_file = open(
            'backdoor_graphs_deg/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphs', 'w')
        train_graph_nodefile = open(
            'backdoor_graphs_deg/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphnodes', 'w')

        rand_backdoor_graph_idx = np.loadtxt(
            'backdoor_graphs_deg/PA/' + str(graph_type) + '_' + str(dataset) + '_' + str(frac) + '_triggersize_' + str(
                num_backdoor_nodes) + '_edgeattach_' + str(K) + '.backdoor_graphs', dtype=int)
        print('#backdoor graphs:', len(rand_backdoor_graph_idx))

    graphs, num_classes, tag2index = load_data(dataset, degree_as_tag)
    train_graphs, test_graphs, test_idx = separate_data(graphs, seed, fold_idx)
    print('#train:', len(train_graphs), '#class:', num_classes)

    num_backdoor_train_graphs = int(frac * len(train_graphs))
    print('#backdoor graphs:', num_backdoor_train_graphs)

    # # Backdoor: target class: target_label
    # # label 1,2,... -> target_label
    train_graphs_target_label_indexes = []
    train_backdoor_graphs_indexes = []

    for graph_idx in range(len(train_graphs)):
        if train_graphs[graph_idx].label == target_label:
            train_graphs_target_label_indexes.append(graph_idx)
        else:
            train_backdoor_graphs_indexes.append(graph_idx)

    print('#train target label:', len(train_graphs_target_label_indexes), '#train backdoor labels:', len(train_backdoor_graphs_indexes))

    rand_backdoor_graph_idx = random.sample(train_backdoor_graphs_indexes, k=num_backdoor_train_graphs) # without replacement

    for idx in rand_backdoor_graph_idx:
        degree_list = []
        select_nodes=[]
        candidate_nodes=[]
        node_query=[]
        num_nodes = torch.max(train_graphs[idx].edge_mat).numpy() + 1
        if num_backdoor_nodes>=num_nodes:
            num_backdoor_nodes=num_nodes
        for node in range(num_nodes):
            degree_list.append((node, len(train_graphs[idx].neighbors[node])))
        # print('degree list:', degree_list)
        sorted_nodes = sorted(degree_list, key=lambda k: k[1], reverse=True)
        add_node=sorted_nodes[0][0]
        select_nodes.append(add_node)
        while(len(select_nodes)<num_backdoor_nodes):
            for node in candidate_nodes:
                if add_node in train_graphs[idx].neighbors[node[0]]:
                    node[2] += 1
            for i in train_graphs[idx].neighbors[add_node]:
                if i not in node_query:
                    node_query.append(i)
                    connect=0
                    for node in select_nodes:
                        if i in train_graphs[idx].neighbors[node]:
                            connect+=1
                    candidate_nodes.append([i,degree_list[i][1],connect])
            candidate_nodes= sorted(candidate_nodes, key= lambda k:(k[2],k[1]),reverse=True)
            add_node=candidate_nodes.pop(0)[0]
            node_query.remove(add_node)
            select_nodes.append(add_node)

        edges = train_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()

        for i in select_nodes:
            for j in select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])

        for e in G_gen.edges:
            if e[0]<len(select_nodes) and e[1]<len(select_nodes):
                edges.append([select_nodes[e[0]], select_nodes[e[1]]])
                edges.append([select_nodes[e[1]], select_nodes[e[0]]])

        train_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        train_graphs[idx].label = target_label
        train_graphs[idx].node_tags = list(dict(train_graphs[idx].g.degree).values())
        train_graphs[idx].node_features = torch.zeros(len(train_graphs[idx].node_tags), len(tag2index))
        train_graphs[idx].node_features[range(len(train_graphs[idx].node_tags)), [tag2index[tag] for tag in train_graphs[idx].node_tags]] = 1

    # train_labels = torch.LongTensor([graph.label for graph in train_graphs])
    # print(train_labels)

    test_graphs_targetlabel_indexes = []
    test_backdoor_graphs_indexes = []
    for graph_idx in range(len(test_graphs)):
        if test_graphs[graph_idx].label != target_label:
            test_backdoor_graphs_indexes.append(graph_idx)
        else:
            test_graphs_targetlabel_indexes.append(graph_idx)
    print('#test target label:', len(test_graphs_targetlabel_indexes), '#test backdoor labels:',
          len(test_backdoor_graphs_indexes))

    for idx in test_backdoor_graphs_indexes:
        degree_list = []
        select_nodes=[]
        candidate_nodes=[]
        node_query=[]
        num_nodes = torch.max(test_graphs[idx].edge_mat).numpy() + 1
        if num_backdoor_nodes>=num_nodes:
            num_backdoor_nodes=num_nodes

        for node in range(num_nodes):
            degree_list.append((node,len(test_graphs[idx].neighbors[node])))

        sorted_nodes = sorted(degree_list, key=lambda k: k[1], reverse=True)
        add_node = sorted_nodes[0][0]
        select_nodes.append(add_node)
        while (len(select_nodes) < num_backdoor_nodes):
            for node in candidate_nodes:
                if add_node in test_graphs[idx].neighbors[node[0]]:
                    node[2] += 1
            for i in test_graphs[idx].neighbors[add_node]:
                if i not in node_query:
                    node_query.append(i)
                    connect = 0
                    for node in select_nodes:
                        if i in test_graphs[idx].neighbors[node]:
                            connect += 1
                    candidate_nodes.append([i, degree_list[i][1], connect])
            candidate_nodes = sorted(candidate_nodes, key=lambda k: (k[2], k[1]), reverse=True)
            add_node = candidate_nodes.pop(0)[0]
            node_query.remove(add_node)
            select_nodes.append(add_node)

        edges = test_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()

        ### Remove existing edges
        for i in select_nodes:
            for j in select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
        # print('after remove:', len(edges))

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            if e[0]<len(select_nodes) and e[1]<len(select_nodes):
                # print([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
                edges.append([select_nodes[e[0]], select_nodes[e[1]]])
                edges.append([select_nodes[e[1]], select_nodes[e[0]]])
        # print('after add:', len(edges))
        # print('new edges:', edges)

        test_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        test_graphs[idx].node_tags = list(dict(test_graphs[idx].g.degree).values())
        test_graphs[idx].node_features = torch.zeros(len(test_graphs[idx].node_tags), len(tag2index))
        test_graphs[idx].node_features[range(len(test_graphs[idx].node_tags)), [tag2index[tag] for tag in test_graphs[idx].node_tags]] = 1

    test_backdoor_graphs = [graph for graph in test_graphs if graph.label != target_label]

    return train_graphs, test_backdoor_graphs

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    # skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    # print(len(train_idx), len(test_idx))

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list, test_idx


