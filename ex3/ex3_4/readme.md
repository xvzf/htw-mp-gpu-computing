# Aufgabe 4

Patrick Plewka (3761940)
Matthias Riegler (3761312)

`idx = blockIdx.x + threadIdx.x * gridDim.x`

`(threadIdx.x * gridDim.x) mod 3 == 0`, da `B == gridDim.x` und `B mod 3 == 0`.

Damit gilt `idx mod 3 == blockIdx.x mod 3`.
