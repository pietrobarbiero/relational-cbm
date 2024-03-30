import numpy as np
from torch import nn
from collections import defaultdict
from torch_geometric import nn as geom_nn

import pytorch_lightning as L

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

# PL callbacks
from torch import Tensor

gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, return_last_hidden = False):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for i in range(len(self.layers)-1):
            layer = self.layers[i]
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        last_hidden = x
        x = self.layers[-1](x)
        if return_last_hidden:
            return x, last_hidden
        return x

class NodeLevelGNN(L.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        print(f'Epoch {self.current_epoch}: train_loss: {loss:.4f} train_acc: {acc:.4f}')
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)
        print(f'val_acc: {acc:.4f}')

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)
        print(f'test_acc: {acc:.4f}')


def train_node_classifier(seed, dataset, **model_kwargs):
    L.seed_everything(seed)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer


    trainer = L.Trainer(
        max_epochs=200,
        enable_progress_bar=True,
    )
    model = NodeLevelGNN(c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
    )
    trainer.fit(model, node_data_loader, node_data_loader)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    x, edge_index = batch.x, batch.edge_index
    model.model.eval()
    _, new_features = model.model(x, edge_index, return_last_hidden = True)
    return model, result, new_features


if __name__ == '__main__':
    name = "CORA"
    n_seeds = 1
    train_size_perc = [1.]
    graph_dataset = torch_geometric.datasets.Planetoid(root='../data/%s' % name, name=name)
    results = []
    res_tab = defaultdict(lambda: np.zeros([n_seeds, 4])) # 4 = len((seed, train_acc, val_acc, test_acc))
    for train_size in train_size_perc:
        if train_size < 1.:
            wh = torch.where(graph_dataset[0].train_mask)[0]
            to_false = wh[:int((len(wh)*(1 - train_size)))]
            graph_dataset[0].train_mask[to_false] = False
        for seed in range(n_seeds):
            # train_node_classifier(dataset=cora_dataset, c_hidden=256, layer_name="GraphConv", num_layers=3, dp_rate_linear=0.5, dp_rate=0.0)
            if name == "CORA":
                _,results, new_features = train_node_classifier(seed=seed, layer_name="GCN", dataset=graph_dataset, c_hidden=16, num_layers=3, dp_rate=0.2)
            torch.save(new_features, "gnn_features_%s_%d_%f.pt" % (name, seed, train_size))
            res_tab[train_size][seed] = (seed, results['train'], results['val'], results['test'])


    for train_size in train_size_perc:
        print("Train size: %f" % train_size)
        print("Avg acc:", np.mean(res_tab[train_size][:,3]))
        print("Std acc:", np.std(res_tab[train_size][:,3]))