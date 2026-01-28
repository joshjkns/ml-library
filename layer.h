#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

typedef struct {
  matrix *w1;
  matrix *s1;
} layer;

void layer_free(layer *layer);
layer *layer_alloc(int sample_count, int input_count, int output_count);

#endif
