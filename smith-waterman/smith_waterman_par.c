#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* Scoring constants */
int MATCH = 3;
int MISMATCH = -3;
int SPACE = -2;

/* Function Prototypes */
int get_num_of_elements_in_stage(int i, int x_len, int y_len);
int max(int a, int b);
int min(int a, int b);
void print_reverse(char* x, int x_len);
void print_matrix(int m, int n, int A[m][n]);
void smith_waterman_forward(char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int i, int j /*int* max_score, int* stage_max, int* max_row, int* max_col */ );
void smith_waterman_backward(char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int* max_row, int* max_col);


int main() {
  char * x = "GGTTGACTAGCTCTGAGCGAGCTAGCACCGTAACGTCACTGACGGTTGACTAGCTCTGAGCGAGCTAGCACCGTAACGTCACTGAC";
  char * y = "TGTTACGGACTGCAGCTCTGAGCGAGCTAGCACCTCAGCAGCATGTTACGGACTGCAGCTCTGAGCGAGCTAGCACCTCAGCAGCA";
  int x_len = strlen(x);
  int m = strlen(x)+1;
  int n = strlen(y)+1;
  int score[m][n];
  int pred[m][n];
  // int stage_max = 0;
  int max_score = 0;
  int max_row, max_col;
  int num_of_stages = (m-2) + (n-2) + 1;
  int start_i, start_j, curr_i, curr_j;
  int i, j;
  int num_of_elements_in_stage;

  // Initialize the first row and first column of matrix to 0
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      score[i][j]=0;
      pred[i][j]=0;
    }
  }
  #pragma omp parallel num_threads(1) \
  default(none) shared(x, y, score, pred, m, n, max_row, max_col, num_of_stages) private(num_of_elements_in_stage, i, j, start_i, start_j, curr_i, curr_j)
  {
    clock_t t;
    t = clock();
    for (i = 0; i < num_of_stages; i++) {
      num_of_elements_in_stage = get_num_of_elements_in_stage(i, m, n);
      if (i < m - 1){
        start_i = i+1;
        start_j = 1;
      } else {
        start_i = m - 1;
        start_j = i - m + 3;
      }
      #pragma omp parallel for
      for (j = 0; j < num_of_elements_in_stage; j++) {
        curr_i = start_i-j;
        curr_j = start_j+j;
        smith_waterman_forward(x, y, m, n, score, pred, curr_i, curr_j);
      }
    }
    t = clock() - t;
    printf("Time: %f\n", (double)t/CLOCKS_PER_SEC);
  }
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      if (score[i][j] > max_score) {
        max_score = score[i][j];
        max_row = i;
        max_col = j;
      }
    }
  }
  smith_waterman_backward(x, y, m, n, score, pred, &max_row, &max_col);

  return 0;
}

// Get number of elements in stage based on stage number and lengths of strings
int get_num_of_elements_in_stage(int i, int x_len, int y_len) {
  int max_dim = max(x_len, y_len);
  int min_dim = min(x_len, y_len);
  if (min_dim > i) {
    return i;
  } else if (max_dim > i) {
    return min_dim - 1;
  } else {
    return 2*min_dim-i+(max_dim-min_dim)-2;
  }
}

/* Finds the max of two integers */
int max(int a, int b) {
  if (a > b) {
    return a;
  }
  return b;
}

/* Find the min of two integers */
int min(int a, int b) {
  if (a < b) {
    return a;
  }
  return b;
}

/* Prints reverse of string */
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

/* Prints matrix */
void print_matrix(int m, int n, int A[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      printf("%d ", A[i][j]);
    }
    printf("\n");
  }
}

/* Forward algorithm to complete score matrix and predecessor matrix */
void smith_waterman_forward(
  char* x, // first string
  char* y, // second string
  int m, // num of rows in matrix
  int n, // num of cols in matrix
  int score[m][n], // score matrix
  int pred[m][n], // predecessor matrix (0 = no pred, 1 = diagonal, 2 = left, 3 = up)
  int i, // current row index
  int j // current col index
) {
  int diagonal, left, up;

  // Calculate diagonal, up, and left
  if (x[i-1] == y[j-1]) {
    diagonal = score[i-1][j-1] + MATCH;
  } else {
    diagonal = score[i-1][j-1] + MISMATCH;
  }
  up = score[i-1][j] + SPACE;
  left = score[i][j-1] + SPACE;
  // Set score & predecessor for current subproblem
  if ((max(0,max(diagonal,max(up,left)))) == 0) {
    score[i][j] = 0;
    pred[i][j] = 0;
  } else if (diagonal >= up && diagonal >= left) {
    score[i][j] = diagonal;
    pred[i][j] = 1;
  } else if (left >= diagonal && left >= up) {
    score[i][j] = left;
    pred[i][j] = 2;
  } else {
    score[i][j] = up;
    pred[i][j] = 3;
  }
}

/* Traces back in pred matrix to get the best local alignment */
void smith_waterman_backward(
  char* x, // first string
  char* y, // second string
  int m, // num of rows in matrix
  int n, // num of cols in matrix
  int score[m][n], // score matrix
  int pred[m][n], // predecessor matrix (0 = no pred, 1 = diagonal, 2 = left, 3 = up)
  int* max_row, // index of row with max score
  int* max_col // index of column with max score
) {
  int alignment_len = 0;
  int i = *max_row;
  int j = *max_col;
  char result_x_alignment[m+n-1];
  char result_y_alignment[m+n-1];
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
