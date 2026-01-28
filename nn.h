#ifndef NN_H
#define NN_H

#include "layer.h"

typedef struct {
  int count;
  layer **layers;
} nn;

nn *nn_alloc(int sample_count, int *topology, int stages);
void nn_free(nn *network);
void nn_forward(nn *network, matrix *input);
double nn_cost(nn *network, matrix *input, matrix *target);
void nn_learn(nn *network, matrix *input, matrix *target, double eps,
              double rate);

#endif
