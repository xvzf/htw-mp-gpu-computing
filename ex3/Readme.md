# Part 1

> GPU: GTX1050Ti

**A**:

| k | #0 | #1 | #2 | #3 | #4 | avg |
|---|---|---|---|---|---|---|
| 0 |  24.097ms | 23.504ms | 25.207ms | 23.891ms | 25.022ms | 24.344ms |
| 1 |  29.079ms | 29.103ms | 27.995ms | 28.601ms | 28.341ms | 28.624ms |
| 2 |  26.427ms | 26.244ms | 26.227ms | 26.391ms | 26.546ms | 26.367ms |

**B**:

| k | #0 | #1 | #2 | #3 | #4 | avg |
|---|---|---|---|---|---|---|
| 0 |  4.352ms | 4.294ms | 4.342ms | 4.282ms | 4.242ms | 4.302ms |
| 1 |  4.247ms | 4.311ms | 4.220ms | 4.212ms | 4.245ms | 4.247ms |
| 2 |  4.358ms | 4.285ms | 4.389ms | 4.372ms | 4.263ms | 4.333ms |


# Part 2

> `N = 36000`

> CPU: M1 Pro, run 6 threads only so the workload get fully scheduled on the performance cores

Measurements without OMP:
```
Compute time 2447.359000ms
Compute time 2233.519000ms
Compute time 2231.694000ms
Compute time 2232.839000ms
Compute time 2231.556000ms
Avg compute time 2275.393400ms
```

with OMP:
```
Compute time 428.901000ms
Compute time 399.094000ms
Compute time 400.846000ms
Compute time 400.473000ms
Compute time 400.540000ms
Avg compute time 405.970800ms
```

Speedup: `5.6` (expected for the 6 CPU cores).

# Part 3

> `N = 20000`

> CPU: Apple M1 Pro 6 Threads

```
Compute time 127.367000ms
Compute time 124.168000ms
Compute time 123.661000ms
Compute time 123.651000ms
Compute time 124.892000ms
Avg compute time 124.747800ms
```


> GPU: GTX 1050Ti
```
Compute duration 58.23ms
Compute duration 58.12ms
Compute duration 58.08ms
Compute duration 58.07ms
Compute duration 58.10ms
Avg compute duration: 58.12ms
```

Speedup: `2.14`
