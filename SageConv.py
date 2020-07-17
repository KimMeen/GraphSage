# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:29:09 2020

@author: Ming Jin

SAGEConv: For full graph training version
SAGEConv_bigraph: For sampling version based on bipartite graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class SAGEConv(nn.Module):
    '''
    This part in algorithm 1 is between the line 3-7
    
    ** Suitbale for full graph training **
        
    Math:
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})
        
    in_feats:
        Input feature size
    out_feats:
        Output feature size
    aggregator_type:
        Aggregator type to use (Mean, GCN, LSTM, Pool)
    activation:
        Applies an activation function to the updated node features
    '''

    def __init__(self, in_feats, out_feats, aggregator_type, activation=None):
        
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats  # "C" in the paper
        self._out_feats = out_feats  # "F" in the paper
        self._activation_func = activation  # "ReLu" and "Softmax" in the paper
        self._aggre_type  = aggregator_type
        
        # aggregator weight and bias
        if aggregator_type != 'gcn':
            # Notice: fc_self is for other aggregators to use in line 5 (Algorithm 1) 
            self.fc_self = nn.Linear(self._in_feats, self._out_feats)  # bias: Optional
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_feats, self._in_feats)
        
        # weight and bias in line 5 (Algorithm 1)
        # GCN aggregator use this as well       
        self.fc_neigh = nn.Linear(self._in_feats, self._out_feats)  # bias: Optional
        
        self.reset_parameters()
       
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        # if self._aggre_type == 'lstm':
        #     self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        
    def forward(self, g, features):
        '''        
        Inputs:
            g: 
                The graph
            features: 
                H^{l}, i.e. Node features with shape [num_nodes, features_per_node]
                
        Returns:
            rst:
                H^{l+1}, i.e. Node embeddings of the l+1 layer (depth) with the 
                shape [num_nodes, hidden_per_node]
                
        Variables:
            msg_func: 
                Message function, i.e. What to be aggregated 
                (e.g. Sending node embeddings)
            reduce_func: 
                Reduce function, i.e. How to aggregate 
                (e.g. Summing neighbor embeddings)
                
        Notice: 'h' means node feature/embedding itself, 'm' means node's mailbox
        '''
        # create an independent instance of the graph to manipulate
        g = g.local_var()
        
        # H^{k-1}_{v}
        h_self = features
        
        # calculate H^{k}_{N(v)} in line 4 of the algorithm 1
        # based on different aggregators
        if self._aggre_type == 'mean':
            g.ndata['h'] = features
            msg_func = fn.copy_src('h', 'm')
            reduce_func = fn.mean('m', 'neigh')
            g.update_all(msg_func, reduce_func)
            # h_neigh is H^{k}_{N(v)}
            h_neigh = g.ndata.pop('neigh')
        elif self._aggre_type == 'gcn':
            # part of equation (2) in the paper
            g.ndata['h'] = features
            msg_func = fn.copy_src('h', 'm')
            reduce_func = fn.sum('m', 'neigh')
            g.update_all(msg_func, reduce_func)
            h_neigh = g.ndata.pop('neigh')
            # H^{k-1}_{v} U H^{k-1}_{u} in equation (2)
            # g.ndata.pop('neigh') represents {H^{k-1}_{u} for u /belongs N(v)}
            # g.dstdata['h'] represents {H^{k-1}_{v}}
            h_neigh = h_neigh + g.ndata.pop('h')
            # divide in_degrees: MEAN() operation in equation (2)
            degs = g.in_degrees().to(features)
            # Notice: h_neigh is more than H^{k}_{N(u)}
            h_neigh = h_neigh / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            g.ndata['h'] = F.relu(self.fc_pool(features))
            msg_func = fn.copy_src('h', 'm')
            reduce_func = fn.max('m', 'neigh')
            g.update_all(msg_func, reduce_func)
            # h_neigh is H^{k}_{N(v)}
            h_neigh = g.ndata.pop('neigh')
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
            
        # calculate H^{k}_{v} in line 5 of the algorithm 1
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
            
        # activation
        if self._activation_func is not None:
            rst = self._activation_func(rst)
        
        # normalization in line 7 of the algorithm 1
        # l2_norm = torch.norm(rst, p=2, dim=1)
        # l2_norm = l2_norm.unsqueeze(1)
        # rst = torch.div(rst, l2_norm)
        
        return rst
        

class SAGEConv_bigraph(nn.Module):
    '''
    This part in algorithm 2 is between the line 10-14
    
    ** Suitbale for mini-batch graph training **
            
    in_feats:
        Input feature size. 
        ** The layer is to be applied on a unidirectional bipartite graph,
        'in_feats' specifies the input feature size on both the source 
        and destination nodes.
        ** If aggregator type is 'gcn', the feature size of source and destination nodes
        are required to be the same.
    out_feats:
        Output feature size
    aggregator_type:
        Aggregator type to use (Mean, GCN, LSTM, Pool)
    activation:
        Applies an activation function to the updated node features
    '''

    def __init__(self, in_feats, out_feats, aggregator_type, activation=None):
        
        super(SAGEConv_bigraph, self).__init__()
        # return a pair of same element if in_feats is not a pair
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = out_feats  # "F" in the paper
        self._activation_func = activation  # e.g. ReLu
        self._aggre_type  = aggregator_type
        
        # aggregator weight and bias
        if aggregator_type != 'gcn':  
            # Notice: fc_self is for other aggregators to use in line 12 (Algorithm 2)
            self.fc_self = nn.Linear(self._in_dst_feats, self._out_feats)  # bias: Optional
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
            self.relu = nn.ReLU()
        
        # weight and bias in line 12 (Algorithm 2)
        # GCN aggregator use this as well
        self.fc_neigh = nn.Linear(self._in_src_feats, self._out_feats)  # bias: Optional
        
        self.reset_parameters()
       
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        # if self._aggre_type == 'lstm':
        #     self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        
    def forward(self, g, features):
        '''        
        Inputs:
            g: 
                The graph
            features: 
                H^{l}, BLOCK.SRC and BLOCK.DST features in tuple with shape
                [N_{src}, D_{in_{src}] and [N_{dst}, D_{in_{dst}]
                where 'D_{in}' is size of input feature
                
        Returns:
            rst:
                H^{l+1}, Node embeddings of the l+1 layer (depth) with the 
                shape [N_{dst}, D_{out}]
                
        Variables:
            msg_func: 
                Message function, i.e. What to be aggregated 
                (e.g. Sending node embeddings)
            reduce_func: 
                Reduce function, i.e. How to aggregate 
                (e.g. Summing neighbor embeddings)
                
        Notice: 'h' means node feature/embedding itself, 'm' means node's mailbox
        '''
        
        # create an independent instance of the graph to manipulate
        g = g.local_var()
        
        # split (feature_src, feature_dst)
        feat_src = features[0]
        feat_dst = features[1]
        
        # H^{k-1}_{u}
        h_self = feat_dst
        
        # calculate H^{k}_{N(u)} in line 11 of the algorithm 2
        # different aggregators: aggregate neighbor (block.src) information
        # in this case, g.srcdata and g.dstdata will be more convenient, they
        # should be identical to g.ndata
        if self._aggre_type == 'mean':
            g.srcdata['h'] = feat_src
            msg_func = fn.copy_src('h', 'm')
            reduce_func = fn.mean('m', 'neigh')
            g.update_all(msg_func, reduce_func)
            # h_neigh is H^{k}_{N(u)}
            h_neigh = g.dstdata['neigh']
        elif self._aggre_type == 'gcn':
            # check whether feat_src and feat_dst has the same shape
            # otherwise we can't sum later
            dgl.utils.check_eq_shape(features)
            # part of equation (2) in the paper
            g.srcdata['h'] = feat_src
            g.dstdata['h'] = feat_dst
            msg_func = fn.copy_src('h', 'm')
            reduce_func = fn.sum('m', 'neigh')
            g.update_all(msg_func, reduce_func)
            h_neigh = g.dstdata['neigh']
            # H^{k-1}_{v} U H^{k-1}_{u} in equation (2)
            # g.dstdata['neigh'] represents BLOCK.DST with aggregation from SRC
            # g.dstdata['h'] represents original BLOCK.DST without aggregation
            h_neigh = h_neigh + g.dstdata['h']
            # divide in_degrees: MEAN() operation in equation (2)
            degs = g.in_degrees().to(feat_dst)
            # Notice: h_neigh is more than H^{k}_{N(u)}
            h_neigh = h_neigh / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            # equation (3) in the paper
            g.srcdata['h'] = self.relu(self.fc_pool(feat_src))
            msg_func = fn.copy_src('h', 'm')
            reduce_func = fn.max('m', 'neigh')
            g.update_all(msg_func, reduce_func)
            # h_neigh is H^{k}_{N(u)}
            h_neigh = g.dstdata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
            
        # calculate H^{k}_{v} in line 11 of the algorithm 2
        # Notice: GCN aggregator is different than in others, see equation (2)
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            # line 12 of the algorithm 2
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
            
        # activation
        if self._activation_func is not None:
            rst = self._activation_func(rst)
        
        # normalization in line 13 of the algorithm 2
        # l2_norm = torch.norm(rst, p=2, dim=1)
        # l2_norm = l2_norm.unsqueeze(1)
        # rst = torch.div(rst, l2_norm)
        
        return rst