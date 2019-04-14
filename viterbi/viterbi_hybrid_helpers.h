#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int max(int a, int b);
float get_rand_float(float min, float max);
void copy_new_to_old(int t, int q, int stage_num, float s1[q], int s2[q], float dp1[t][q], int dp2[t][q]);
void fix_stage(int n, int lp, int q, int t, float s1[q], int s2[q], float dp[t][q], int Y[t], float A[q][q], float B[q][n]);
bool is_parallel(int t, int q, int stage_idx, float s[q], float dp[t][q]);
void print_arr(int t, int X[t]);
void print_fl_arr(int t, float X[t]);
void print_matrix(int m, int n, int A[m][n]);
void convert_to_log_prob(int m, int n, float matrix[m][n]);
void convert_array_to_log_prob(int n, float arr[n]);

/* Finds the max of two integers */
int max(int a, int b) {
  if (a > b) {
    return a;
  }
  return b;
}

/* Random float between min and max */
float get_rand_float(float min, float max) {
    return  (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min ;
}

/* Copy new stage to old */
void copy_new_to_old(int t, int q, int stage_num, float s1[q], int s2[q], float dp1[t][q], int dp2[t][q]) {
  for (int i = 0; i < q; i++) {
    dp1[stage_num][i] = s1[i];
    dp2[stage_num][i] = s2[i];
  }
}

/* Calculates new array of values for fixup phase */
void fix_stage(int n, int lp, int q, int t, float s1[q], int s2[q], float dp[t][q], int Y[t], float A[q][q], float B[q][n]) {
  for (int i = 0; i < q; i++) {
    float max = -INFINITY;
    int arg_max = -1;
    double curr_log_prob;
    #pragma omp parallel for
    for (int j = 0; j < q; j++) {
      curr_log_prob = dp[lp-1][j] + A[j][i] + B[i][Y[lp]];
      // Update max and curr_max if needed
      if (curr_log_prob > max) {
        max = curr_log_prob;
        arg_max = j;
      }
    }
    s1[i] = max;
    s2[i] = arg_max;
  }
}

/* Checks if two stages are parallel */
bool is_parallel(int t, int q, int stage_idx, float s[q], float dp[t][q]) {
  for (int i = 1; i < q; i++) {
    if ((s[i-1]-dp[stage_idx][i-1]) != (s[i]-dp[stage_idx][i])) {
      return false;
    }
  }
  return true;
}

/* Prints int array */
void print_arr(int t, int X[t]) {
  for (int i=0; i<t; i++) {
    printf("%d ", X[i]);
  }
  printf("\n");
}

/* Prints float array */
void print_fl_arr(int t, float X[t]) {
  for (int i=0; i<t; i++) {
    printf("%f ", X[i]);
  }
  printf("\n");
}

/* Prints matrix */
void print_fl_matrix(int m, int n, float A[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }
}

/* Prints matrix */
void print_matrix(int m, int n, int A[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      printf("%d ", A[i][j]);
    }
    printf("\n");
  }
}

/* Convert a matrix to log probability */
void convert_to_log_prob(int m, int n, float matrix[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      matrix[i][j] = (float)log(matrix[i][j]);
    }
  }
}

/* Convert to log probability */
void convert_array_to_log_prob(int n, float arr[n]) {
  for (int i = 0; i < n; i++) {
    arr[i] = (float)log(arr[i]);
  }
}
