# R-GCN
An implementation of R-GCN for entity classification task by tensorflow.

## Reference
**R-GCN**: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103v4.pdf) (Code: https://github.com/MichSchli/RelationPrediction)   

## Results (10 runs)                    
|         |    **aifb**   |    **bgs**    |   **mutag**   |  
|    --   |      --       |      --       |      --       |  
|**R-GCN**| 0.958 (0.022) | 0.852 (0.016) | 0.703 (0.038) |     

```
python Run_RGCN.py --dataset aifb --h_dim 16 --n_B 0 --dropout 0.0 --l2 0.0 --l_r 1e-2
```
```
python Run_RGCN.py --dataset bgs --h_dim 16 --n_B 0 --dropout 0.0 --l2 5e-4 --l_r 1e-2
```
```
python Run_RGCN.py --dataset mutag --h_dim 16 --n_B 100 --dropout 0.2 --l2 5e-4 --l_r 1e-2
```
