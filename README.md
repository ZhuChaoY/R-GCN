# R-GCN
An implementation of R-GCN for entity classification task by tensorflow.

## Reference
**R-GCN**: Modeling Relational Data with Graph Convolutional Networks (https://github.com/MichSchli/RelationPrediction)   

## Operation
```
python Run_RGCN.py --dataset aifb --n_B 0 --h_dim 16 --dropout 0.0 --l2 0.0 --l_r 1e-2 --epoches 200 --earlystop 3
```

```
python Run_RGCN.py --dataset mutag --n_B 40 --h_dim 16 --dropout 0.0 --l2 5e-4 --l_r 1e-2 --epoches 200 --earlystop 3
```

## Results (10 runs)                    
|         |    **aifb**   |    **bgs**    |   **mutag**   |  
|    --   |      --       |      --       |      --       |  
|**R-GCN**| 0.967 (0.017) |               | 0.687 (0.024) |     
