# R-GCN
An implementation of R-GCN for entity classification task by tensorflow.

## Reference
**R-GCN**: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103v4.pdf) (Code: https://github.com/MichSchli/RelationPrediction)   

## Results (10 runs)                    
|         |    **aifb**   |    **bgs**    |   **mutag**   |  
|    --   |      --       |      --       |      --       |  
|**R-GCN**| 0.967 (0.017) | 0.852 (0.016) | 0.725 (0.017) |     

```
python Run_RGCN.py --dataset aifb --n_B 0 --h_dim 16 --l2 0.0 --l_r 1e-2
```
```
python Run_RGCN.py --dataset bgs --n_B 0 --h_dim 16 --l2 5e-4 --l_r 1e-2
```
```
python Run_RGCN.py --dataset mutag --n_B 100 --h_dim 16 --l2 5e-4 --l_r 1e-2
```
