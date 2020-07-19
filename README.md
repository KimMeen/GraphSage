# GraphSage


A PyTorch implementation of the paper https://www-cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf



## Requirements

Pytorch >=1.1.0

DGL: 0.4.3.post2



## Results

The node classification results include two parts: 

+ Full graph training on Cora citation dataset
+ Minibatch training on Reddit dataset



### Full graph training

Run with following to train a GraphSage network on the Cora dataset:

```
python train_full_cora.py
```

**Notice:** This version not performs neighbor sampling (i.e. Algorithm 1 in the paper) so we feed the model with the entire graph and corresponding feature matrix.



+ GraphSage-Mean: ~ 80.4%
+ GraphSage-GCN: ~ 83.4%
+ GraphSage-Pool: ~ 72.5%



### Minibatch training

<img src=".\docs\DGL_Minibatch.png" alt="DGL_MINIBATCH" style="zoom:75%;" />



Run with following to train a GraphSage network on the Reddit dataset:

```
python train_sampling_reddit.py
```

**Notice:** This version performs neighbor sampling in a layer-wise way (i.e. Algorithm 2 in the paper) so we feed the model with blocks (undirected bipartite graph).



+ GraphSage-Mean: ~ 96.33%
+ GraphSage-GCN: ~ 94.53%
+ GraphSage-Pool: ~ 88.35%



## To-do:

+ LSTM aggregator

+ Minibatch training: Inductive graph splitting
+ Unsupervised training