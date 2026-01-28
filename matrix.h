#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

typedef struct {
  int rows;
  int cols;
  double *data;
} matrix;

matrix *mat_alloc(int rows, int cols);
void mat_free(matrix *mat);
void mat_copy(matrix *src, matrix *dst);
void mat_fill(matrix *mat, double val);
void mat_rand(matrix *mat);
void mat_print(matrix *mat);
void mat_mul(matrix *out, matrix *a, bool trans_a, matrix *b, bool trans_b);
void mat_mul_subset(matrix *out, matrix *a, bool trans_a, matrix *b,
                    bool trans_b, int active_cols);
void mat_add(matrix *out, matrix *a, matrix *b);
void mat_sub(matrix *out, matrix *a, matrix *b);
double mat_sum(matrix *mat);
void mat_sigmoid(matrix *mat);
void mat_sigmoid_subset(matrix *mat, int active_cols);
double mat_mse(matrix *target, matrix *pred);
double mat_mse_subset(matrix *target, matrix *pred, int active_cols);
double mat_cost(matrix *weights, matrix *input, matrix *target,
                matrix *scratch);
void mat_learn(matrix *weights, matrix *input, matrix *target, matrix *scratch,
               double eps, double rate);
#endif
