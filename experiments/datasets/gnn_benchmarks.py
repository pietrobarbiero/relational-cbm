from torch_geometric.datasets import Planetoid
from torch.utils.data import Dataset, TensorDataset
import torch
import numpy as np



def gnn_benchmark_dataset(name, perc_train = 1., pretrain_seed = -1):

    if name in {'CORA', 'CITESEER', 'PubMed'}:
        dataset = Planetoid(root='../data/%s' % name, name=name)
        g = dataset[0]
        X = g.x
        manifold_ids =  np.array(g.edge_index.T).astype('str')
        num_classes = torch.max(g.y) +1

        if perc_train < 1.:
            wh = torch.where(g.train_mask)[0]
            to_false = wh[:int((len(wh)*(1 - perc_train)))]
            g.train_mask[to_false] = False


        queries_train = ["class%d(%d)" % (c,i)  for i in range(len(X)) if g.train_mask[i] for c in range(num_classes)]
        queries_valid = ["class%d(%d)" % (c,i)  for i in range(len(X))  if g.val_mask[i] for c in range(num_classes)]
        queries_test = ["class%d(%d)" % (c,i) for i in range(len(X))  if g.test_mask[i] for c in range(num_classes)]

        labels_train = [int(c==g.y[i]) for i in range(len(X)) if g.train_mask[i] for c in range(num_classes)]
        labels_valid = [int(c==g.y[i]) for i in range(len(X)) if g.val_mask[i] for c in range(num_classes)]
        labels_test =  [int(c==g.y[i]) for i in range(len(X)) if g.test_mask[i] for c in range(num_classes)]


        X = X.unsqueeze(0)
        if pretrain_seed>-1:
            X = torch.load("gnn_features_%s_%d_%f.pt" % (name, pretrain_seed, perc_train))
            X = X.unsqueeze(0)
        train_data = TensorDataset(X, torch.tensor(labels_train).unsqueeze(0), torch.tensor(labels_train).unsqueeze(0))
        valid_data = TensorDataset(X, torch.tensor(labels_valid).unsqueeze(0), torch.tensor(labels_valid).unsqueeze(0))
        test_data = TensorDataset(X, torch.tensor(labels_test).unsqueeze(0), torch.tensor(labels_test).unsqueeze(0))

        return X, (train_data, queries_train), (valid_data,queries_valid), (test_data, queries_test), num_classes, manifold_ids

    else:
        raise Exception(" Dataset %s unknown" % name)



if __name__ == '__main__':

    X, (train_data, queries_train), (valid_data, queries_valid), (test_data, queries_test), manifold_ids = gnn_benchmark_dataset("CORA")
    print()