# Exercise 1
> Note: I ran the experiments on an Apple M1 processor therefore I limited the number of threads to 4 in order to only schedule the workload on the performance cores

## Test runs
> Processor: M1, Memory: 8GB

|   N   |    T(4)     |    T(4)    | S(4) |
|-------|-------------|------------|------|
| 40000 |  5664.71 ms | 1488.86 ms | 3.80 |
| 25000 |  2198.10 ms |  576.74 ms | 3.81 |
| 10000 |   354.71 ms |   93.35 ms | 3.80 |
|  5000 |    88.31 ms |   23.23 ms | 3.80 |
|  1000 |     6.67 ms |    1.69 ms | 3.95 |
|   500 |     2.11 ms |    0.64 ms | 3.30 |
|   250 |     0.55 ms |    0.24 ms | 2.29 |
|   100 |     0.09 ms |    0.12 ms |   /  |
|    50 |     0.02 ms |    0.11 ms |   /  |
|     4 |     0.00 ms |    0.10 ms |   /  |

### Conclusions
The speedup factor remains rather constant across when the workload is large. Starting with N < 500 we can see the speedup factor going down.
Lower N (especially e.g. N=50 or N=4) show the overhead of the dataset split-up and initialisation of openmp.