#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int max(int a, int b);
int min(int a, int b);
void print_reverse(char* x, int x_len);
void print_matrix(int m, int n, int A[m][n]);
char* generate_sequence(int n);
int get_num_of_elements_in_stage(int i, int x_len, int y_len);
int get_num_of_stages(int m, int n);
int get_start_row(int j,int m);
int get_start_col(int j,int m);
void copy_new_soln(int m, int n, int start_i, int start_j, int new_score[m][n], int new_pred[m][n], int score[m][n], int pred[m][n]);

/*
 * Function: max
 * --------------------
 *  Returns max of two integers
 *
 *  a: first integer
 *  b: second integer
 *
 *  returns: the larger of a and b
 */
int max(int a, int b) {
  if (a > b) {
    return a;
  }
  return b;
}

/*
 * Function: min
 * --------------------
 *  Returns min of two integers
 *
 *  a: first integer
 *  b: second integer
 *
 *  returns: the smaller of a and b
 */
int min(int a, int b) {
  if (a < b) {
    return a;
  }
  return b;
}

/*
 * Function: print_reverse
 * --------------------
 *  Prints reverse of string
 *
 *  x: string to print in reverse
 *  x_len: length of x
 */
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

/*
 * Function: print_matrix
 * --------------------
 *  Prints matrix
 *
 *  m: num of rows in A
 *  n: num of cols in A
 *  A: matrix to print
 */
void print_matrix(int m, int n, int A[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      printf("%d ", A[i][j]);
    }
    printf("\n");
  }
}

/*
 * Function: generate_sequence
 * --------------------
 *  Randomly generates a DNA sequence of length n
 *
 *  n: length of
 *
 *  returns: randomly generated DNA sequence of length n
 */
char* generateSequence(
  int n
) {
  char* seq = malloc(sizeof(char) * (n+1));;
  char seqChars[] = {'A','C','G','T'};
  for (int i=0; i<n; i++) {
    seq[i] = seqChars[rand()%4];
  }
  seq[n] = '\0';
  return seq;
}

//

/*
 * Function: get_num_of_elements_in_stage
 * --------------------
 *  Get number of elements in stage based on stage number and lengths of strings
 *
 *  i: the stage index
 *  x_len: length of first string x
 *  y_len: length of second string y
 *
 *  returns: number of elements in stage i
 */
int get_num_of_elements_in_stage(int i, int x_len, int y_len) {
  int max_dim, min_dim;
  if (x_len > y_len) {
    max_dim = x_len;
    min_dim = y_len;
  } else {
    max_dim = y_len;
    min_dim = x_len;
  }
  if (min_dim > i) {
    return i + 1;
  } else if (max_dim > i) {
    return min_dim;
  } else {
    return 2*min_dim-i+(max_dim-min_dim)-1;
  }
}

/*
 * Function: get_num_of_stages
 * --------------------
 *  Get number of stages in a matrix
 *
 *  m: num of rows in matrix
 *  n: num of cols in matrix
 *
 *  returns: number of stages in a m by n matrix
 */
int get_num_of_stages(int m, int n) {
  return (m-2) + (n-2) + 1;
}

/*
 * Function: get_start_row
 * --------------------
 *  Get start row index for a stage
 *
 *  j: stage index
 *  m: num of rows in matrix
 *
 *  returns: start row index for stage j
 */
int get_start_row(
  int j,
  int m
) {
  if (j < m - 1){
    return j+1;
  }
  return m - 1;
}

/*
 * Function: get_start_col
 * --------------------
 *  Get start col index for a stage
 *
 *  j: stage index
 *  m: num of rows in matrix
 *
 *  returns: start col index for stage j
 */
int get_start_col(
  int j,
  int m
) {
  if (j < m - 1){
    return 1;
  }
  return j - m + 3;
}

/*
 * Function: copy_new_soln
 * --------------------
 *  Copies a new solution vectors to an old solution vector in score and pred
 *
 *  m: num of rows in matrix
 *  n: num of cols in matrix
 *  start_i: starting row index of solution vector
 *  start_j: starting col index of solution vector
 *  new_score: score matrix with new solution vector
 *  new_pred: predecessor matrix with new solution vector
 *  score: score matrix with old solution vector
 *  pred: predecessor matrix with old solution vector
 */
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
