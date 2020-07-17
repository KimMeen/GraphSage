'''
Previous DGL official example WITHOUT using:
    
    + MultiLayerNeighborSampler()
    
    + and NodeDataLoader()

Build version:
    
    + PyTorch 1.1.0
    
    + DGL 0.4.3.post2

referred from https://zhuanlan.zhihu.com/p/142205899 
'''

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.nn import SAGEConv
import time
from dgl.data import RedditDataset
import tqdm


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        """
        论文算法2的第1至7行
        
        g 为 DGLGraph；
        fanouts 为采样节点的数量，实验使用 10,25，指一阶邻居采样 10 个，二阶邻居采样 25 个。
        
        e.g. 一个种子的长度是1000，就是一个batch的size, 1000个一个batch， 采样10个邻居，得
             到10000边9640个点，再采样25个邻居，得到241000个边，105693个点，Blocks里面是两个子图
        """
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        # 在若干个二部图串联在一起的前提下（即blocks），思路是从左向右采样
        # 最终得到若干个blocks，包含了batch_size个nodes(即seeds)若干个hops聚合操作所需要的全部nodes
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts: 
            if fanout is None:
                frontier = dgl.transform.in_subgraph(self.g, seeds)
            else:
                # sample_neighbors 可以对每一个种子的节点进行邻居采样并返回相应的子图，即frontier
                # replace=True 表示用采样后的邻居节点代替所有邻居节点 （具体的还是要查一下）
                frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # # to_black操作是把将采样的子图转换为适合计算的二部图
            # 这里特殊的地方在于block.srcdata中的id是包含了dstnodeid的
            block = dgl.to_block(frontier, seeds)
            # 获取新图的源节点作为种子节点，为下一层作准备
            # 即将本层的scrnode作为下一层的seeds来采样邻居节点
            # 之所以是从 src 中获取种子节点，是因为采样操作相对于聚合操作来说是一个逆向操作
            seeds = block.srcdata[dgl.NID]
            # 把这一层放在最前面
            # 假设有两层(K=2)，那么最后的blocks=[block1, block0], 序号表示loop
            blocks.insert(0, block)
            
        return blocks
    
# GraphSAGE 的代码实现
class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        '''
        论文算法2的第8至16行
        
        在转化为若干二部图的前提下（即blocks），思路是从右向左聚合，最终得到batch_size个nodes的embedding
        blocks 是我们采样获得的二部图, e.g. blocks = [block1, block0]
        x 为最外(K)层节点的特征, e.g. 2-hop nodes with their features
        x 包含了所有需要用到的节点的特征, 
        e.g. h = [block1.dstnodes && block1.srcnodes 即 block0.dstnodes && block0.srcnodes]
        
        h维度变化的例子：
        loop1: h.shape: torch.Size([105727, 602])
        loop2: h.shape: torch.Size([9644, 16])
        return h.shape: torch.Size([1000, 41])
        '''
        h = x  # h.shape = [num_nodes_K, hidden_per_node]
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # 每一阶的节点里面都包含了他的dst节点在序列的最前面，方便计算。
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))  # h.shape = [num_nodes_K-1, hidden_per_node]
            # 最后一层不加activation和dropout
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h  # h.shape = [batch_size, n_classes]

    def inference(self, g, x, batch_size, device):
        '''
        inference 用于evaluation，用的是完全图
        
        g 是完整的graph
        x 是g的特征，即g.ndata['features']
        batch_size 是每次放入layer中计算的batch大小
        
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.

        '''
        # nodes = th.arange(g.number_of_nodes())
        # for l, layer in enumerate(self.layers):
        #     y = th.zeros(g.number_of_nodes(), 
        #                   self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
        #     for start in tqdm.trange(0, len(nodes), batch_size):
        #         end = start + batch_size
        #         batch_nodes = nodes[start:end]
        #         block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
        #         input_nodes = block.srcdata[dgl.NID]
        #         h = x[input_nodes].to(device)
        #         h_dst = h[:block.number_of_dst_nodes()]
        #         h = layer(block, (h, h_dst))
        #         if l != len(self.layers) - 1:
        #             h = self.activation(h)
        #             h = self.dropout(h)
        #         y[start:end] = h.cpu()
        #     x = y
        # return y
        
        # 按照聚合顺序，逐层构建g中全部节点的representations
        for l, layer in enumerate(self.layers):
            # y用于存储预测的g的全部节点的k层hidden representation
            y = th.zeros(g.number_of_nodes(), 
                          self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            
            # 用dataloader按照batch_size遍历g中的全部节点
            # sampler不指定采样邻居数量，则返回1-hop全部邻居的一个block
            sampler = NeighborSampler(g, [None])
            dataloader = DataLoader(
                dataset=th.arange(g.number_of_nodes()),
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for blocks in tqdm.tqdm(dataloader):
                # len(blocks) = 1
                block = blocks[0]
                
                # input_nodes.shape: torch.Size([138408])
                # output_nodes.shape: torch.Size([1000])
                input_nodes = block.srcdata[dgl.NID]
                output_nodes = block.dstdata[dgl.NID]
                # h表示output_nodes的1-hop邻居的特征，即h_src            
                h = x[input_nodes].to(device)
                # h_dst表示output_nodes的特征
                h_dst = h[:block.number_of_dst_nodes()]
                # 放入k层对应的layer中预测g的全部节点的k层hidden representation
                h = layer(block, (h, h_dst))
                # 如果是最后一层，不要activation和dropout
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                # 里层looping来收集预测，直到dataloader把g的全部节点都抽了一遍
                y[output_nodes] = h.cpu()
            # 将预测的g的全部节点的k层hidden representation作为新的features给下层用
            x = y
        # 直到最后一层, return y.shape: torch.Size([232965, 41])
        # 232965是Reddit全部节点数目，41是n_classes
        return y        
   
    
def compute_acc(pred, labels):
    """
    计算训练集上的准确率
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    评估模型，调用 model 的 inference 函数处理验证集和测试集
    inputs = g.ndata['features'], labels = th.LongTensor(data.labels)
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    将一组节点的特征和标签复制到 GPU 上。
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

### entry point

# 参数设置
gpu = 0
num_epochs = 20
num_hidden = 16
num_layers = 2
fan_out = '10,25'
batch_size = 1000
log_every = 20  # 记录日志的频率
eval_every = 10
lr = 0.003
dropout = 0.5
num_workers = 0  # 用于采样进程的数量

if gpu >= 0:
    device = th.device('cuda:%d' % gpu)
else:
    device = th.device('cpu')
    
# load reddit data
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
features = th.Tensor(data.features)
in_feats = features.shape[1]
labels = th.LongTensor(data.labels)
n_classes = data.num_labels
# Construct graph
g = dgl.graph(data.graph.all_edges())
g.ndata['features'] = features

### 开始训练
train_nid = th.LongTensor(np.nonzero(train_mask)[0])
val_nid = th.LongTensor(np.nonzero(val_mask)[0])
train_mask = th.BoolTensor(train_mask)
val_mask = th.BoolTensor(val_mask)

# Create sampler
sampler = NeighborSampler(g, [int(fanout) for fanout in fan_out.split(',')])

# Create PyTorch DataLoader for constructing blocks
# collate_fn 参数指定了 sampler，可以对 batch 中的节点进行采样
dataloader = DataLoader(
    dataset=train_nid.numpy(),
    batch_size=batch_size,
    collate_fn=sampler.sample_blocks,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers)

# Define model and optimizer
model = GraphSAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout)
model = model.to(device)
loss_fcn = nn.CrossEntropyLoss()
loss_fcn = loss_fcn.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
avg = 0
iter_tput = []
for epoch in range(num_epochs):
    tic = time.time()

    for step, blocks in enumerate(dataloader):
        tic_step = time.time()
        
        # len(blocks): 等于K，即layers数目，表示有多少个子二部图
        
        # 假设K=2，那么：
        # blocks[0] 即邻居采样算法中的block1，表示 1-hop & 2-hop 子二部图
        # block[1] 即邻居采样算法中的block0， 表示 seeds & 1-hop 子二部图
        
        # 接下来看一下各个block的src和dst的nodes数目 （batch_size=1000, fanout='10,25'）
        # blocks[0].srcdata[dgl.NID].shape: torch.Size([106421])
        # blocks[0].dstdata[dgl.NID].shape: torch.Size([9670])
        # blocks[1].srcdata[dgl.NID].shape: torch.Size([9670])
        # blocks[1].dstdata[dgl.NID].shape: torch.Size([1000])
        
        # 输入的节点特征是seeds的K-th邻居节点的特征（最右边）
        # 它里面包括了所有blocks的节点，详见算法2的第3和第5行
        input_nodes = blocks[0].srcdata[dgl.NID]
        # seeds即一个batch内需要计算embedding的节点，即种子节点
        seeds = blocks[-1].dstdata[dgl.NID]
        
        # input_nodes.shape: torch.Size([106421])
        # seeds.shape: torch.Size([1000])
        
        # Load the input features as well as output labels
        batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)
        
        # batch_inputs.shape: torch.Size([106421, 602])
        # batch_labels.shape: torch.Size([1000])
            
        # Compute loss and prediction
        # model返回seeds个，也就是batch_size个nodes的embedding
        batch_pred = model(blocks, batch_inputs)
        # CrossEntropyLoss包含了softmax，与label比较计算negative log loss
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_tput.append(len(seeds) / (time.time() - tic_step))
        if step % log_every == 0:
            acc = compute_acc(batch_pred, batch_labels)
            gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
            print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

    toc = time.time()
    print('Epoch Time(s): {:.4f}'.format(toc - tic))
    if epoch >= 5:
        avg += toc - tic
    if epoch % eval_every == 0 and epoch != 0:
        eval_acc = evaluate(model, g, g.ndata['features'], labels, val_mask, batch_size, device)
        print('Eval Acc {:.4f}'.format(eval_acc))
        
eval_acc = evaluate(model, g, g.ndata['features'], labels, val_mask, batch_size, device)
print('Eval Acc {:.4f}'.format(eval_acc))
print('Avg epoch time: {}'.format(avg / (epoch - 4)))