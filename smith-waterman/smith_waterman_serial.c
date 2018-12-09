#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Scoring constants */
int MATCH = 2;
int MISMATCH = -1;
int SPACE = -1;

/* Function Prototypes */
int max(int a, int b);
void print_reverse(char* x, int x_len);
void print_matrix(int m, int n, int A[m][n]);
void smith_waterman_forward(char* x, char* y, int x_len, int y_len, int score[x_len+1][y_len+1], int pred[x_len+1][y_len+1], int* max_row, int* max_col);
void smith_waterman_backward(char* x, char* y, int x_len, int y_len, int score[x_len+1][y_len+1], int pred[x_len+1][y_len+1], int* max_row, int* max_col);

/* Finds the max of two integers */
int max(int a, int b) {
  if (a > b) {
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
  int x_len, // first string length
  int y_len, // second string length
  int score[x_len+1][y_len+1], // score matrix
  int pred[x_len+1][y_len+1], // predecessor matrix (0 = no pred, 1 = diagonal, 2 = left, 3 = up)
  int* max_row, // index of row with max score
  int* max_col // index of column with max score
) {
  int i,j;
  int diagonal, left, up;
  int max_score = 0;
  *max_row = 0;
  *max_col = 0;
  // Initialize the first row and first column of matrix to 0
  for (i=0; i<=x_len; i++) {
    for (j=0; j<=y_len; j++) {
      score[i][j]=0;
      pred[i][j]=0;
    }
  }

  // Fill out matrices
  for (i=1; i<=x_len; i++) {
    for (j=1; j<=y_len; j++) {
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

      // Update max if needed
      if (score[i][j] > max_score) {
        max_score = score[i][j];
        *max_row = i;
        *max_col = j;
      }
    }
  }
}

/* Traces back in pred matrix to get the best local alignment */
void smith_waterman_backward(
  char* x, // first string
  char* y, // second string
  int x_len, // first string length
  int y_len, // second string length
  int score[x_len+1][y_len+1], // score matrix
  int pred[x_len+1][y_len+1], // predecessor matrix (0 = no pred, 1 = diagonal, 2 = left, 3 = up)
  int* max_row, // index of row with max score
  int* max_col // index of column with max score
) {
  int alignment_len = 0;
  int i = *max_row;
  int j = *max_col;
  char result_x_alignment[x_len+y_len+1];
  char result_y_alignment[x_len+y_len+1];
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

char * generateSequence(
  int n // the length of the sequence
) {

}

int main(int argc, const char* argv[]) {
  char * x = "AGTACAGA";
  char * y = "ACGCACTA";
  int x_len = strlen(x);
  int y_len = strlen(y);
  int score[x_len+1][y_len+1];
  int pred[x_len+1][y_len+1];
  int max_row;
  int max_col;
  clock_t t;
  t = clock();
  smith_waterman_forward(x, y, x_len, y_len, score, pred, &max_row, &max_col);
  t = clock() - t;
  printf("Time: %f\n", (double)t/CLOCKS_PER_SEC);
  smith_waterman_backward(x, y, x_len, y_len, score, pred, &max_row, &max_col);
  print_matrix(x_len+1,y_len+1,score);
  return 0;
}
