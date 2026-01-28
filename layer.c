#include "layer.h"
#include <stdlib.h>

void layer_free(layer *layer) {
  mat_free(layer->w1);
  mat_free(layer->s1);
  free(layer);
}

layer *layer_alloc(int sample_count, int input_count, int output_count) {
  layer *l = malloc(sizeof(layer));
  l->w1 = mat_alloc(input_count, output_count);
  mat_rand(l->w1);

  l->s1 = mat_alloc(sample_count, output_count + 1);

  // fill last column (bias) with all 1.0
  for (int i = 0; i < sample_count; i++) {
    l->s1->data[i * l->s1->cols + output_count] = 1.0;
  }

  return l;
}
