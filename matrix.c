#include "matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

matrix *mat_alloc(int rows, int cols) {
  matrix *mat = malloc(sizeof(matrix));
  mat->data = malloc(sizeof(double) * rows * cols);
  mat->rows = rows;
  mat->cols = cols;
  return mat;
}

void mat_free(matrix *mat) {
  free(mat->data);
  free(mat);
}

void mat_copy(matrix *src, matrix *dst) {
  assert(src->rows == dst->cols);
  assert(src->cols == dst->cols);

  for (int i = 0; i < src->rows * src->cols; i++) {
    dst->data[i] = src->data[i];
  }
}

void mat_fill(matrix *mat, double val) {
  for (int i = 0; i < mat->rows * mat->cols; i++) {
    mat->data[i] = val;
  }
}

void mat_rand(matrix *mat) {
  for (int i = 0; i < mat->rows * mat->cols; i++) {
    double r = (double)rand() / (double)RAND_MAX;
    mat->data[i] = r;
  }
}

void mat_print(matrix *mat) {
  for (int y = 0; y < mat->rows; y++) {
    for (int x = 0; x < mat->cols; x++) {
      printf("%7.2f", mat->data[y * mat->cols + x]);
    }
    printf("\n");
  }
  printf("\n");
}

void mat_mul(matrix *out, matrix *a, bool trans_a, matrix *b, bool trans_b) {

  int a_rows = trans_a ? a->cols : a->rows;
  int a_cols = trans_a ? a->rows : a->cols;
  int b_rows = trans_b ? b->cols : b->rows;
  int b_cols = trans_b ? b->rows : b->cols;

  assert(a_cols == b_rows);
  assert(out->rows == a_rows);
  assert(out->cols == b_cols);

  for (int y = 0; y < a->rows; y++) {
    for (int x = 0; x < b->cols; x++) {
      double sum = 0;
      for (int k = 0; k < a->cols; k++) {

        int index_a = trans_a ? (k * a->cols + y) : (y * a->cols + k);
        int index_b = trans_b ? (x * b->cols + k) : (k * b->cols + x);

        sum += a->data[index_a] * b->data[index_b];
      }
      out->data[y * b->cols + x] = sum;
    }
  }
}

void mat_mul_subset(matrix *out, matrix *a, bool trans_a, matrix *b,
                    bool trans_b, int active_cols) {
  int a_rows = trans_a ? a->cols : a->rows;
  int a_cols = trans_a ? a->rows : a->cols;
  int b_rows = trans_b ? b->cols : b->rows;
  int b_cols = trans_b ? b->rows : b->cols;

  assert(a_cols == b_rows);
  assert(out->rows == a_rows);
  assert(out->cols >= active_cols);

  for (int y = 0; y < a_rows; y++) {
    for (int x = 0; x < active_cols; x++) {
      double sum = 0;
      for (int k = 0; k < a_cols; k++) {
        int index_a = trans_a ? (k * a->cols + y) : (y * a->cols + k);
        int index_b = trans_b ? (x * b->cols + k) : (k * b->cols + x);

        sum += a->data[index_a] * b->data[index_b];
      }

      out->data[y * out->cols + x] = sum;
    }
  }
}

void mat_sigmoid(matrix *mat) {
  for (int i = 0; i < mat->rows * mat->cols; i++) {
    double sigmoid = 1.0 / (1.0 + exp(-mat->data[i]));
    mat->data[i] = sigmoid;
  }
}

void mat_sigmoid_subset(matrix *mat, int active_cols) {
  for (int y = 0; y < mat->rows; y++) {
    for (int x = 0; x < active_cols; x++) {
      double sigmoid = 1.0 / (1.0 + exp(-mat->data[y * mat->cols + x]));
      mat->data[y * mat->cols + x] = sigmoid;
    }
  }
}

double mat_sum(matrix *mat) {
  double sum = 0;
  for (int i = 0; i < mat->rows * mat->cols; i++) {
    sum += mat->data[i];
  }
  return sum;
}

void mat_add(matrix *dst, matrix *a, matrix *b) {
  assert(a->rows == b->rows && b->rows == dst->rows);
  assert(a->cols == b->cols && b->cols == dst->cols);

  for (int i = 0; i < a->rows * a->cols; i++) {
    dst->data[i] = a->data[i] + b->data[i];
  }
}

void mat_sub(matrix *dst, matrix *a, matrix *b) {
  assert(a->rows == b->rows && b->rows == dst->rows);
  assert(a->cols == b->cols && b->cols == dst->cols);

  for (int i = 0; i < a->rows * a->cols; i++) {
    dst->data[i] = a->data[i] - b->data[i];
  }
}

double mat_mse(matrix *target, matrix *pred) {
  assert(target->rows == pred->rows);
  assert(target->cols == pred->cols);

  int n = target->rows * target->cols;

  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum +=
        (target->data[i] - pred->data[i]) * (target->data[i] - pred->data[i]);
  }
  return sum / n;
}

double mat_mse_subset(matrix *target, matrix *pred, int active_cols) {
  assert(target->rows == pred->rows);

  double sum = 0;
  for (int y = 0; y < target->rows; y++) {
    for (int x = 0; x < active_cols; x++) {
      double error =
          target->data[y * target->cols + x] - pred->data[y * pred->cols + x];
      sum += error * error;
    }
  }

  return sum / (target->rows * active_cols);
}

double mat_cost(matrix *weights, matrix *input, matrix *target,
                matrix *scratch) {
  mat_mul(scratch, input, false, weights, false);
  mat_sigmoid(scratch);
  return mat_mse(target, scratch);
}

void mat_learn(matrix *weights, matrix *input, matrix *target, matrix *scratch,
               double eps, double rate) {

  for (int i = 0; i < weights->rows * weights->cols; i++) {
    double old = weights->data[i];

    double cost = mat_cost(weights, input, target, scratch);

    weights->data[i] += eps;
    double cost_new = mat_cost(weights, input, target, scratch);

    weights->data[i] = old;
    weights->data[i] -= rate * (cost_new - cost) / eps;
  }
}
