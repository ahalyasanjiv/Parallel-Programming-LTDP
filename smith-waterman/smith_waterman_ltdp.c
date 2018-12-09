#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

int MATCH = 2;
int MISMATCH = -1;
int SPACE = -1;

int max(int a, int b);
void print_matrix(int m, int n, int A[m][n]);
void print_reverse(char* x, int x_len);
bool is_parallel( int m, int n, int start_i, int start_j, int new_score[m][n], int score[m][n]);
int get_start_row( int j, int m );
int get_start_col( int j, int m );
void calculate_element(int up, int diagonal, int left, int * score, int * pred);
int get_num_of_stages(int m, int n);
int get_num_of_elements_in_stage(int i, int m, int n);
void forward( char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int *max_score_arr, int *max_row_arr, int *max_col_arr, int block_size, int num_stages, int num_teams, int num_threads );
void fixup( char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int *max_score_arr, int *max_row_arr, int *max_col_arr, int block_size, int num_stages, int num_teams, int num_threads );
void backward( char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int max_row, int max_col);
void smith_waterman(char* x, char* y);

/* Finds the max of two integers */
int max(int a, int b) {
  if (a > b) {
    return a;
  }
  return b;
}

void print_matrix(int m, int n, int A[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      printf("%d ", A[i][j]);
    }
    printf("\n");
  }
}

void print_reverse(char* x, int x_len) {
  char* new_str = malloc(x_len);
  int i;
  for (i=0; i<x_len; i++) {
    new_str[i] = x[x_len-i-1];
  }
  new_str[i] = '\0';
  printf("%s\n", new_str);
  free(new_str);
}

bool is_parallel(
  int m,
  int n,
  int start_i,
  int start_j,
  int new_score[m][n],
  int score[m][n]
) {
  bool parallel = true;

  for (int i = 0; start_j+i < n-1 && start_i - i > 1; i++) {
    if ((new_score[start_i-i][start_j+i] - score[start_i-i][start_j+i]) != (new_score[start_i-i][start_j+i] - score[start_i-i-1][start_j+i+1])) {
      parallel = false;
    }
    else {
      parallel = true;
    }
  }

  return parallel;
}

int get_start_row(
  int j,
  int m
) {
  if (j < m - 1){
    return j+1;
  }
  return m - 1;
}

int get_start_col(
  int j,
  int m
) {
  if (j < m - 1){
    return 1;
  }
  return j - m + 3;
}

void copy_new_soln(
  int m,
  int n,
  int start_i,
  int start_j,
  int new_score[m][n],
  int new_pred[m][n],
  int score[m][n],
  int pred[m][n]
) {
  int i = start_i;
  for (int j = start_j; j < n && i > 0; j++) {
    score[i][j] = new_score[i][j];
    pred[i][j] = new_pred[i][j];
    i--;
  }
}

void calculate_element(int up, int diagonal, int left, int * score, int * pred) {
  // Set score & predecessor for current subproblem
  if ((max(0,max(diagonal,max(up,left)))) == 0) {
    *score = 0;
    *pred = 0;
  } else if (diagonal >= up && diagonal >= left) {
    *score = diagonal;
    *pred = 1;
  } else if (left >= diagonal && left >= up) {
    *score = left;
    *pred = 2;
  } else {
    *score = up;
    *pred = 3;
  }
}

int get_num_of_stages(int m, int n) {
  return (m-2) + (n-2) + 1;
}

// Get number of elements in stage based on stage number and lengths of strings
int get_num_of_elements_in_stage(int i, int m, int n) {
  int max_dim, min_dim;
  if (m > n) {
    max_dim = m;
    min_dim = n;
  } else {
    max_dim = n;
    min_dim = m;
  }
  if (min_dim > i) {
    return i;
  } else if (max_dim > i) {
    return min_dim - 1;
  } else {
    return 2 * min_dim - i + (max_dim - min_dim) - 2;
  }
}

void forward(
  char* x,
  char* y,
  int m,
  int n,
  int score[m][n],
  int pred[m][n],
  int *max_score_arr,
  int *max_row_arr,
  int *max_col_arr,
  int block_size,
  int num_stages,
  int num_teams,
  int num_threads
) {
  omp_set_num_threads(4);
  int max_threads = omp_get_max_threads();

#pragma omp parallel num_threads(max_threads)
  {
    int tid = omp_get_thread_num();
    int team_num = floor((num_teams/num_threads) * (tid/num_threads));
    int lp = team_num * block_size;
    int rp = lp + block_size;

    // Create arrays to store initial read values for up, diagonal, and left
    int up[n];
    int diagonal[n];
    int left[n];
    int start_row, start_col, curr_row, curr_col;

#pragma omp barrier
    for (int i = lp; i < rp; i++)
    {
      start_row = get_start_row(i,m);
      start_col = get_start_col(i,m);
      curr_row = start_row - (tid % num_threads);
      curr_col = start_col + (tid % num_threads);
#pragma omp barrier
      for (int j = 0; j < block_size; j++)
      {
        if (curr_col < n && curr_row > 0) {
          if (x[curr_row-1] == y[curr_col-1]) {
            diagonal[curr_col] = score[curr_row-1][curr_col-1] + MATCH;
          } else {
            diagonal[curr_col] = score[curr_row-1][curr_col-1] + MISMATCH;
          }
          up[curr_col] = score[curr_row-1][curr_col] + SPACE;
          left[curr_col] = score[curr_row][curr_col-1] + SPACE;
          curr_col += num_threads;
          curr_row -= num_threads;
        }
      }
      curr_row = start_row - (tid % num_threads);
      curr_col = start_col + (tid % num_threads);
#pragma omp barrier
      for (int j = 0; j < block_size; j++) {
        if (curr_col < n && curr_row > 0) {
          calculate_element(up[curr_col],diagonal[curr_col],left[curr_col],&score[curr_row][curr_col],&pred[curr_row][curr_col]);
          if (score[curr_row][curr_col] > max_score_arr[tid]) {
            max_score_arr[tid] = score[curr_row][curr_col];
            max_row_arr[tid] = curr_row;
            max_col_arr[tid] = curr_col;
          }
          curr_col += num_threads;
          curr_row -= num_threads;
        }
      }
    }
  }
}

void fixup(
  char* x,
  char* y,
  int m,
  int n,
  int score[m][n],
  int pred[m][n],
  int *max_score_arr,
  int *max_row_arr,
  int *max_col_arr,
  int block_size,
  int num_stages,
  int num_teams,
  int num_threads
) {
  int max_threads = omp_get_max_threads();
  bool converged = false;
  bool conv[num_teams];
  for (int i = 1; i < num_teams; i++) {
    conv[i] = false;
  }

  int new_score[m][n];
  int new_pred[m][n];

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      new_score[i][j] = 0;
      new_pred[i][j] = 0;
    }
  }

  do {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int team_num = floor((num_teams/num_threads) * (tid/num_threads));
      int lp = team_num * block_size + block_size;
      int rp = lp + block_size;
      int up[n];
      int diagonal[n];
      int left[n];
      int start_row, start_col, curr_row, curr_col;
      bool parallel = false;
      #pragma omp barrier
      for (int i = lp; i < rp; i++)
      {
        start_row = get_start_row(i,m);
        start_col = get_start_col(i,m);
        curr_row = start_row - (tid % num_threads);
        curr_col = start_col + (tid % num_threads);
      #pragma omp barrier
        for (int j = 0; j < block_size; j++)
        {
          if (curr_col < n && curr_row > 0) {
            if (x[curr_row-1] == y[curr_col-1]) {
              diagonal[curr_col] = score[curr_row-1][curr_col-1] + MATCH;
            } else {
              diagonal[curr_col] = score[curr_row-1][curr_col-1] + MISMATCH;
            }
            up[curr_col] = score[curr_row-1][curr_col] + SPACE;
            left[curr_col] = score[curr_row][curr_col-1] + SPACE;
            curr_col += num_threads;
            curr_row -= num_threads;
          }
        }
        curr_row = start_row - (tid % num_threads);
        curr_col = start_col + (tid % num_threads);
        #pragma omp barrier
        for (int j = 0; j < block_size; j++) {
          if (curr_col < n && curr_row > 0) {
            calculate_element(up[curr_col],diagonal[curr_col],left[curr_col],&new_score[curr_row][curr_col],&new_pred[curr_row][curr_col]);
            if (new_score[curr_row][curr_col] > max_score_arr[tid]) {
              max_score_arr[tid] = new_score[curr_row][curr_col];
              max_row_arr[tid] = curr_row;
              max_col_arr[tid] = curr_col;
            }
            curr_col += num_threads;
            curr_row -= num_threads;
          }
        }
        #pragma omp barrier
        parallel = is_parallel(m,n,start_row,start_col,new_score,score);
        if (parallel) {
          conv[i] = true;
          break;
        }
        copy_new_soln(m,n,start_row,start_col,new_score,new_pred,score,pred);
      }
    }
    converged = true;
    for (int i = 1; i < num_threads; i++) {
      if (!conv[i]) {
        converged = false;
        break;
      }
    }
  } while (converged);
  print_matrix(m,n,score);
  printf("\n");
  print_matrix(m,n,new_score);

}

void backward(
  char* x,
  char* y,
  int m,
  int n,
  int score[m][n],
  int pred[m][n],
  int max_row,
  int max_col
) {
  int alignment_len = 0;
  char result_x_alignment[m+n-1];
  char result_y_alignment[m+n-1];
  int i = max_row;
  int j = max_col;
  // Keep tracing back from element with max_score until we hit a 0
  while (pred[i][j] != 0) {
    switch(pred[i][j]) {
      case 1: // diagonal
        result_x_alignment[alignment_len] = x[i-1];
        result_y_alignment[alignment_len] = y[j-1];
        i--;
        j--;
        break;
      case 2: // left
        result_x_alignment[alignment_len] ='-';
        result_y_alignment[alignment_len] = y[j-1];
        j--;
        break;
      case 3: // up
        result_x_alignment[alignment_len] = x[i-1];
        result_y_alignment[alignment_len] = '-';
        i--;
        break;
    }
    alignment_len++;

  }
  print_reverse(result_x_alignment,alignment_len);
  print_reverse(result_y_alignment,alignment_len);
}

void smith_waterman(char* x, char* y) {
  int m = strlen(x) + 1;
  int n = strlen(y) + 1;
  int score[m][n];
  int pred[m][n];
  int num_cores = 4;

  int *max_score_arr = (int *)malloc(sizeof(int)*num_cores);
  int *max_row_arr = (int *)malloc(sizeof(int)*num_cores);
  int *max_col_arr = (int *)malloc(sizeof(int)*num_cores);

  // Initialize matrices
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      score[i][j] = 0;
      pred[i][j] = 0;
    }
  }
  int num_teams = 2; // for coarse-grain parallelism
  int num_threads = num_cores / num_teams; // for fine-grain parallelism
  int num_stages = get_num_of_stages(m,n);
  int block_size = (int)ceil((float)num_stages / num_teams);
  forward(x,y,m,n,score,pred,max_score_arr,max_row_arr,max_col_arr,block_size,num_stages,num_teams,num_threads);

  // rp[tid]nt_matrix(m,n,score);
  // printf("\n");
  // print_matrix(m,n,pred);
  fixup(x,y,m,n,score,pred,max_score_arr,max_row_arr,max_col_arr,block_size,num_stages,num_teams-1,num_threads);
  int max_row = 0;
  int max_col = 0;
  int max_score = 0;
  for (int i = 0; i < num_cores; i++) {
    if (max_score_arr[i] >= max_score) {
      max_row = max_row_arr[i];
      max_col = max_col_arr[i];
    }
  }
  backward(x,y,m,n,score,pred,max_row,max_col);
}

int main() {
  char * x = "AGTACAGA";
  char * y = "ACGCACTA";
  smith_waterman(x,y);
  return 0;
}
