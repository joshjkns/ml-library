#include "nn.h"
#include <stdlib.h>

nn *nn_alloc(int sample_count, int *topology, int stages) {
  nn *network = malloc(sizeof(nn));
  network->count = stages - 1;
  network->layers = malloc(sizeof(layer *) * network->count);

  int input_count = topology[0];

  for (int i = 0; i < network->count; i++) {
    int output_count = topology[i + 1];

    network->layers[i] = layer_alloc(sample_count, input_count, output_count);

    input_count = output_count + 1;
  }

  return network;
}

void nn_free(nn *network) {
  for (int i = 0; i < network->count; i++) {
    layer_free(network->layers[i]);
  }
  free(network);
}

void nn_forward(nn *network, matrix *input) {
  matrix *curr = input;

  for (int i = 0; i < network->count; i++) {
    layer *l = network->layers[i];

    mat_mul_subset(l->s1, curr, false, l->w1, false, l->w1->cols);

    mat_sigmoid_subset(l->s1, l->w1->cols);

    curr = l->s1;
  }
}

double nn_cost(nn *network, matrix *input, matrix *target) {
  nn_forward(network, input);
  layer *last_layer = network->layers[network->count - 1];

  return mat_mse_subset(target, last_layer->s1, last_layer->w1->cols);
}

void nn_learn(nn *network, matrix *input, matrix *target, double eps,
              double rate) {
  for (int l = 0; l < network->count; l++) {
    matrix *w = network->layers[l]->w1;

    for (int i = 0; i < w->rows * w->cols; i++) {
      double old = w->data[i];

      double cost = nn_cost(network, input, target);

      w->data[i] += eps;

      double cost_new = nn_cost(network, input, target);

      w->data[i] = old;

      double derivative = (cost_new - cost) / eps;
      w->data[i] -= rate * derivative;
    }
  }
}
