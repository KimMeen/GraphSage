# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 19:51:00 2020

@author: Ming Jin

Minibatch training (Algorithm 2) on Reddit dataset 
with the concept of bipartite graph

Build version:
    
    + PyTorch 1.1.0
    
    + DGL 0.4.3.post2
"""

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from dgl.data import RedditDataset
import tqdm

from SageConv import SAGEConv_bigraph


class NeighborSampler(object):
    """
    Line 1 to 7 of the algorithm 2 in the paper
    
    g: DGLGraph
    fanouts: sampling neighbors, len(fanouts) is the number of returned blocks
    
    seeds: minibatched nodes that we we want to generate representations for
    
    Return:
        blocks: a set of bipartite graphs in the order of [block K, ..., block 0]
                block 0: Seeds(dst) and 1-hop neighbors(src)
                block K: K-hop neighbors(dst) and (K+1)-hop neighbors(src)
    
    Example: 
        If we set batch_size=1000, then the size of the seeds will be 1000;
        If fanouts = '10,25', then we sample 10 neighbors per seed and will 
        have 1-hop neighbors; After this, we sample 25 neighbors per 1-hop
        neighbors to get 2-hop neighbors; Seeds and 1-hop neighbors form the
        block 0, 1-hop neighbors and 2-hop neighbors form the block 1;
        
    """
    def __init__(self, g, fanouts):
        
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        
        # Based on the idea of bipartite graph, sampling will be performed
        # from the LHS to RHS, i.e. Seeds to their neighbors
        # Notice: Outmost neighbors, i.e. blocks[0].srcdata[dgl.NID], includes 
        # all of the nodes that will be needed for k-hops aggregation of the 
        # bathc_size seeds
        
        seeds = torch.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts: 
            if fanout is None:
                # This will be used during the inference
                # return the subgraph contains ALL 1-hop neighbors
                frontier = dgl.transform.in_subgraph(self.g, seeds)
            else:
                # sample_neighbors() samples 'fanout' neighbors of 'seeds' on 'g'
                # TODO: the meaning of replace=True needs to check
                frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True) 
            # to_block() converts 'frontier' to a bipartite graph with dst as 'seeds'
            # 'include_dst_in_src=True' means to include DST nodes in SRC nodes
            # Since DST nodes are included in SRC nodes, we could fetch
            # the DST node features from the SRC nodes features (this is why I wrote
            # such a notice at the begining of this method).
            block = dgl.to_block(frontier, seeds, include_dst_in_src=True)
            # assign the SRC of the current block as the DST of next block
            seeds = block.srcdata[dgl.NID]
            # store blocks with the stack structure
            # if there are two layers (K=2), then blocks = [block 1, block 0]
            # the block_id represents loops
            blocks.insert(0, block)
            
        return blocks


class GraphSAGE(nn.Module):
    """
    Minibatch-based SAGE network
    
    Line 8 to 16 of the algorithm 2 in the paper
    
    Parameters:
        blocks: a set of bipartite graphs in the order of [block K, ..., block 0]
        x: feature matrix with shape [block0.DSTNODES, NumFeats]
    """
    
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 aggregator_type,
                 dropout = 0.5):
        
        super(GraphSAGE, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        # first layer
        self.layers.append(SAGEConv_bigraph(in_feats, n_hidden, aggregator_type, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv_bigraph(n_hidden, n_hidden, aggregator_type, activation))
        # last layer
        self.layers.append(SAGEConv_bigraph(n_hidden, n_classes, aggregator_type, None))
        
    def forward(self, blocks, x):
        
        # Based on the idea of bipartite graph, aggregation will be performed
        # layer-wise from the RHS to LHS, i.e. Outmost neighbors to seeds
        # input x with the shape [block0.DSTNODES, NumFeats]
        # return h with the shape [seeds(batch_size), n_classes]
        # Example:
        #       h维度变化的例子：
        #       loop1: h.shape: torch.Size([105727, 602])
        #       loop2: h.shape: torch.Size([9644, 16])
        #       return h.shape: torch.Size([1000, 41])
        
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # h: [BLOCK.DST, BLOCK.SRC]
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))
            # we don't need activation and dropout for the last layer
            # activation has been included inside of the SAGEConv
            if l != len(self.layers) - 1:
                h = self.dropout(h)
        return h
    
    def inference(self, g, x, batch_size, device):

        # Inference with the model on full neighbors (i.e. without neighbor sampling)
        # g: the entire graph
        # x: feature matrix of the entire graph
        # batch_size: we can't feed all node to a layer at once so we need batch_size
        
        # P.S. During inference with sampling, multi-layer blocks are very inefficient
        # because lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer-by-layer
        # The nodes on each layer are of course splitted in batches.
        
        # Predict node representations on graph g layer-by-layer based on the 
        # aggregation order we performed on forward()
        for l, layer in enumerate(self.layers):
            
            # we use 'y' to store the k-th layer predictions of nodes on 'g'
            y = torch.zeros(g.number_of_nodes(), 
                          self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            
            # go through nodes in 'g' batch-by-batch with 'dataloader'
            # we don't need to assign how many neighbors we want to sample cuz
            # WE WANT ALL OF NEIGHBORS OF EACH NODE
            sampler = NeighborSampler(g, [None])
            dataloader = DataLoader(
                dataset=torch.arange(g.number_of_nodes()),
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=True,
                drop_last=False)

            for blocks in tqdm.tqdm(dataloader):
                # Notice: len(blocks) = 1
                block = blocks[0]
                # Example:
                #   input_nodes.shape: torch.Size([138408])
                #   output_nodes.shape: torch.Size([1000])
                input_nodes = block.srcdata[dgl.NID]
                output_nodes = block.dstdata[dgl.NID]
                # h: feature matrix of BLOCK.SRC, which is 1-hop neighbors of the output_nodes
                h = x[input_nodes].to(device)
                # h_dst: feature matrix of BLOCK.DST, which equals to the number of batch_size
                h_dst = h[:block.number_of_dst_nodes()]
                # feed into the corresponding layer to get k-th layer predictions
                # of this batch of nodes, i.e. output_nodes
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.dropout(h)
                # inner-looping to collect k-th layer predictions of ALL nodes in 'g'
                y[output_nodes] = h.cpu()
            # k-th layer predictions will act as the features of (k+1)-th layer
            x = y
        # outer-looping goes through layers to generate the final node embeddings of 'g'
        # Example:
        #   return y.shape: torch.Size([232965, 41])
        #   232965 is NumNodes，41 is n_classes
        return y

def compute_acc(pred, labels):
    """
    Return the accuracy of prediction given the labels. 
    """
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by 'val_nid'
    g : The entire graph.
    inputs : The features of all the nodes, i.e. g.ndata['features']
    labels : The labels of all the nodes, i.e. torch.LongTensor(data.labels)
    val_nid : the node Ids for validation
    batch_size : Number of nodes to compute at the same time
    device : The GPU device to evaluate on
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels


'''
Entry point
'''

# Parameters
gpu = 0
aggregation = 'pool'
num_epochs = 20
num_hidden = 16
num_layers = 2  # n_layers >= 2
fan_out = '10,25'
batch_size = 1000
log_every = 20  # frequency of printout
eval_every = 10
lr = 0.003
dropout = 0.5

if gpu >= 0:
    device = torch.device('cuda:%d' % gpu)
else:
    device = torch.device('cpu')

# Reddit dataset

# NumNodes: 232965
# NumEdges: 114848857
# NumFeats: 602
# NumClasses: 41
# NumTrainingSamples: 153431
# NumValidationSamples: 23831
# NumTestSamples: 55703

data = RedditDataset(self_loop=True)
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask
features = torch.Tensor(data.features)
labels = torch.LongTensor(data.labels)
in_feats = features.shape[1]
n_classes = data.num_labels

# Construct graph
g = dgl.graph(data.graph.all_edges())
g.ndata['features'] = features
g.ndata['labels'] = labels

# get different node IDs
# Examples:
#   train_nid.shape: 153431
#   val_nid.shape: 23831
#   test_nid.shape: 55703   
train_nid = torch.LongTensor(np.nonzero(train_mask)[0])
val_nid = torch.LongTensor(np.nonzero(val_mask)[0])
test_nid = torch.LongTensor(np.nonzero(test_mask)[0])

# Create training sampler
sampler = NeighborSampler(g, [int(fanout) for fanout in fan_out.split(',')])

# Create PyTorch DataLoader for constructing blocks
# collate_fn <-- sampler，which could sample k-hops neighbors for seeds in a batch
dataloader = DataLoader(
    dataset=train_nid.numpy(),
    batch_size=batch_size,
    collate_fn=sampler.sample_blocks,
    shuffle=True,
    drop_last=False)

# Define model and optimizer
model = GraphSAGE(in_feats, 
                  num_hidden, 
                  n_classes, 
                  num_layers, 
                  nn.ReLU(), 
                  aggregation, 
                  dropout)
model = model.to(device)
loss_fcn = nn.CrossEntropyLoss()
loss_fcn = loss_fcn.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
avg = 0
iter_tput = []
for epoch in range(num_epochs):
    tic = time.time()
    # loop over nodes in the training graph
    # return a list of blocks for each batch of seed nodes
    # len(blocks): should equals to num_layers
    # If K=2 then:
    #    blocks[0] is block 1 in sample_blocks(), represents 1-hop & 2-hop bigraph
    #    blocks[1] is block 0 in sample_blocks(), represents seeds & 1-hop bigraph
    # Example (if batch_size=1000, fanout='10,25'):
    #    blocks[0].srcdata[dgl.NID].shape: torch.Size([106421])
    #    blocks[0].dstdata[dgl.NID].shape: torch.Size([9670])
    #    blocks[1].srcdata[dgl.NID].shape: torch.Size([9670])
    #    blocks[1].dstdata[dgl.NID].shape: torch.Size([1000])
    for step, blocks in enumerate(dataloader):
        
        tic_step = time.time()
        
        # As we mentioned before, input_nodes contains all nodes we need
        # This could also be found in line 3 and 5 in algorithm 2 
        input_nodes = blocks[0].srcdata[dgl.NID]
        # seeds is a batch of nodes that we want to calculate their embeddings
        seeds = blocks[-1].dstdata[dgl.NID]
        # Example: 
        #   input_nodes.shape: torch.Size([106421])
        #   seeds.shape: torch.Size([1000])
        
        # Load the input features as well as output labels
        # Example:
        #   batch_inputs.shape: torch.Size([106421, 602])
        #   batch_labels.shape: torch.Size([1000])
        batch_inputs, batch_labels = load_subtensor(g, seeds, input_nodes, device)
        
        # Compute loss and prediction
        # batch_pred with the shape [batch_size, n_classes]
        batch_pred = model(blocks, batch_inputs)
         
        optimizer.zero_grad()
        
        # Notice: CrossEntropyLoss contains softmax() and negative log loss
        loss = loss_fcn(batch_pred, batch_labels)
        loss.backward()
        optimizer.step()

        iter_tput.append(len(seeds) / (time.time() - tic_step))
        if step % log_every == 0:
            # training accuracy
            acc = compute_acc(batch_pred, batch_labels)
            # GPU mem usage
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            # printout
            print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
    
    # Epoch time calculation
    toc = time.time()
    print('Epoch Time(s): {:.4f}'.format(toc - tic))
    if epoch >= 5:
        avg += toc - tic
    # Evaluate on both val_set and test_set
    if epoch % eval_every == 0 and epoch != 0:
        eval_acc = evaluate(model, g, g.ndata['features'], labels, val_nid, batch_size, device)
        print('Eval Acc {:.4f}'.format(eval_acc))
        test_acc = evaluate(model, g, g.ndata['features'], labels, test_nid, batch_size, device)
        print('Test Acc: {:.4f}'.format(test_acc))

test_acc = evaluate(model, g, g.ndata['features'], labels, test_nid, batch_size, device)
print('Test Acc: {:.4f}'.format(test_acc))
print('Avg epoch time: {}'.format(avg / (epoch - 4)))