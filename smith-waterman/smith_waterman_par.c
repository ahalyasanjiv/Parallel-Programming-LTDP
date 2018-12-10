#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers.h"



/* Scoring constants */
int MATCH = 2;
int MISMATCH = -1;
int SPACE = -1;

/* Function Prototypes */
void smith_waterman_forward(char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int i, int j /*int* max_score, int* stage_max, int* max_row, int* max_col */ );
void smith_waterman_backward(char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int* max_row, int* max_col);

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
  printf("Local alignment for reference: ");
  print_reverse(result_x_alignment,alignment_len);
  printf("Local alignment for query: ");
  print_reverse(result_y_alignment,alignment_len);
}

int main() {
  srand(time(NULL));
  int ref_len,query_len;
  printf("===========================================================\n"
         "SMITH WATERMAN SERIAL ALGORITHM\n");
  printf("Computing the local alignment for two random DNA sequences.\n");
  printf("===========================================================\n");
  printf("Enter the string size for the reference string: ");
  scanf("%d",&ref_len);
  printf("Enter the string size for the query string: ");
  scanf("%d",&query_len);
  char* ref = generateSequence(ref_len);
  char* query =
  generateSequence(query_len);
  printf("Reference string: %s\nQuery string: %s\n",ref,query);

  int m = ref_len+1;
  int n = query_len+1;
  int score[m][n];
  int pred[m][n];
  // int stage_max = 0;
  int max_score = 0;
  int max_row, max_col;
  int num_of_stages = m+n-3;
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
  clock_t t;
  t = clock();
  for (i = 0; i < num_of_stages; i++) {
    num_of_elements_in_stage = get_num_of_elements_in_stage(i, m-1, n-1);
    if (i < m - 1){
      start_i = i+1;
      start_j = 1;
    } else {
      start_i = m - 1;
      start_j = i - m + 3;
    }
    #pragma omp parallel for num_threads(4)
    for (j = 0; j < num_of_elements_in_stage; j++) {
      curr_i = start_i-j;
      curr_j = start_j+j;
      smith_waterman_forward(ref, query, m, n, score, pred, curr_i, curr_j);
    }
  }
  t = clock() - t;
  printf("Time taken for forward phase: %f seconds\n", (double)t/CLOCKS_PER_SEC);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      if (score[i][j] > max_score) {
        max_score = score[i][j];
        max_row = i;
        max_col = j;
      }
    }
  }
  smith_waterman_backward(ref, query, m, n, score, pred, &max_row, &max_col);
  return 0;
}
