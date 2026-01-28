#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"

int main(void) {
  srand(time(NULL));

  matrix *input = mat_alloc(4, 3);
  double in_data[] = {0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1};
  for (int i = 0; i < 12; i++)
    input->data[i] = in_data[i];

  matrix *target = mat_alloc(4, 1);
  double tar_data[] = {0, 1, 1, 1};
  for (int i = 0; i < 4; i++)
    target->data[i] = tar_data[i];

  int topo[] = {3, 2, 1};
  int topo_count = sizeof(topo) / sizeof(topo[0]);

  nn *network = nn_alloc(4, topo, topo_count);

  double eps = 1e-1;
  double rate = 1e-1;
  int epochs = 100000;

  printf("Initial Cost: %f\n", nn_cost(network, input, target));

  for (int i = 0; i <= epochs; i++) {
    nn_learn(network, input, target, eps, rate);

    if (i % 10000 == 0) {
      printf("Epoch %d, Cost: %f\n", i, nn_cost(network, input, target));
    }
  }

  nn_forward(network, input);

  layer *last_layer = network->layers[network->count - 1];

  for (int i = 0; i < 4; i++) {
    double in1 = input->data[i * 3];
    double in2 = input->data[i * 3 + 1];
    double out = last_layer->s1->data[i * last_layer->s1->cols];

    printf("Input: [%.0f, %.0f] -> Predicted: %.4f (Expected: %.0f)\n", in1,
           in2, out, tar_data[i]);
  }

  nn_free(network);
  mat_free(input);
  mat_free(target);

  return 0;
}
