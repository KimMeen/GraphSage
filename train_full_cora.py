# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:49:04 2020

@author: Ming Jin

Full graph training (Algorithm 1) on Cora dataset 
For a large graph like Reddit, this approach will compromise so that we need sampling

** For simplicity, I haven't adapt to CUDA for this script **

Build version:
    
    + PyTorch 1.1.0
    
    + DGL 0.4.3.post2
"""

import torch
import torch.nn as nn
import networkx as nx
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

import time
import numpy as np

from SageConv import SAGEConv


class GraphSAGE(nn.Module):
    '''
    Full graph training SAGE network
    (This version is much simpler than the sampling one)
    '''
    
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 aggregator_type,
                 dropout = 0.5):
        
        super(GraphSAGE, self).__init__()   
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        # first layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, activation))
        # last layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, None))
        
    def forward(self, g, features):
        # Similar to GCN, 'g' is the entire graph, 'features' are node features
        # returned h with the shape [num_nodes, n_classes]
        h = features
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            # we don't need activation and dropout for the last layer
            if l != len(self.layers) - 1:
                h = self.dropout(h)
        return h
 
# create a network instance to train/evaluate on Cora
model = GraphSAGE(1433,
                  16,
                  7,
                  2,  # n_layers >= 2
                  nn.ReLU(),
                  "pool")


def load_cora_data():
    '''
    Cora dataset function
    '''
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    # graph preprocess and calculate normalization factor
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    # return graph, node features, labels, and training mask
    return g, features, labels, mask, val_mask, test_mask



### train a 2-layer GraphSage on Cora dataset

g, features, labels, mask, val_mask, test_mask = load_cora_data()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
loss_fcn = nn.CrossEntropyLoss()

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        # run pred on all nodes
        logits = model(graph, features)
        # but only evaluate on val or test nodes
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)  # predicted class index
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

dur = []

for epoch in range(200):
    
    model.train()
    
    if epoch >=3:
        t0 = time.time()
        
    optimizer.zero_grad()
    # feed model with all nodes and features
    # logits with the shape [num_nodes, n_class]
    logits = model(g, features)
    # but only train it on training nodes
    # Notice: we haven't calculate loss on val/test nodes
    loss = loss_fcn(logits[mask], labels[mask])

    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)
    
    if epoch % 1 == 0:
        acc = evaluate(model, g, features, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))

test_acc = evaluate(model, g, features, labels, test_mask)
print("\nTest accuracy {:.2%}".format(test_acc))

