# ml library in c

a basic ml library in c.

currently only sigmoid function, no relu.

basic structure - neural network made up of layers which are made up of 2 matrices (being a weight matrix and an output (scratchpad) matrix).

compile command: `gcc main.c matrix.c layer.c nn.c -o main`

run command: `.\main`

## inspiration

watched like 20 mins of [this vid](https://www.youtube.com/watch?v=hL_n_GljC0I) and tried it myself.

mine is pretty minimal compared to that but will continue to add to this when i have time - more so a project to relearn c and use some of the math knowledge i learnt before i forget it.

## goals

optimised mat_mul for transposed matrices with cache locality.

use mnist or other dataset.

xavier initialisation alongside Relu.

backpropagation.

cross-entropy-loss instead of mse
- softmax
- regularisation

optimise the loops
- simd (cuda??? or maybe openmp)

better initialisation of neural networks for testing.

tryna keep the code looking clean.
