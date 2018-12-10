#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int max(int a, int b);
float getRandFloat(float min, float max);
void print_arr(int t, int X[t]);
void print_fl_arr(int t, float X[t]);
void print_fl_matrix(int m, int n, float A[m][n]);
void print_matrix(int m, int n, int A[m][n]);
void convertToLogProb(int m, int n, float matrix[m][n]);
void convertArrayToLogProb(int n, float arr[n]);

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
 * Function: getRandFloat
 * --------------------
 *  Generates a random float in specified range.
 *
 *  min: lower bound of range for random float
 *  max: upper bound of range for random float
 *
 *  returns: a random float between min and max
 */
float getRandFloat(float min, float max) {
    return  (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min ;
}

/*
 * Function: print_arr
 * --------------------
 *  Prints an int array
 *
 *  t: size of array to print
 *  X: int array to print
 */
void print_arr(int t, int X[t]) {
  for (int i=0; i<t; i++) {
    printf("%d ", X[i]);
  }
  printf("\n");
}

/*
 * Function: print_fl_arr
 * --------------------
 *  Prints a float array
 *
 *  t: size of array to print
 *  X: float array to print
 */
void print_fl_arr(int t, float X[t]) {
  for (int i=0; i<t; i++) {
    printf("%f ", X[i]);
  }
  printf("\n");
}

/*
 * Function: print_fl_arr
 * --------------------
 *  Prints a float matrix
 *
 *  m: num of row in matrix
 *  n: num of cols in matrix
 *  A: float matrix to print
 */
void print_fl_matrix(int m, int n, float A[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }
}

/*
 * Function: print_fl_arr
 * --------------------
 *  Prints an int matrix
 *
 *  m: num of row in matrix
 *  n: num of cols in matrix
 *  A: int matrix to print
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
 * Function: convertToLogProb
 * --------------------
 *  Applies natural log to each element in a float matrix
 *
 *  m: num of row in matrix
 *  n: num of cols in matrix
 *  A: float matrix to convert
 *
 *  returns: resulting matrix after applying natural log to each element of A
 */
void convertToLogProb(int m, int n, float A[m][n]) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      A[i][j] = (float)log(A[i][j]);
    }
  }
}

/*
 * Function: convertArrayToLogProb
 * --------------------
 *  Applies natural log to each element in a float array
 *
 *  n: size of array
 *  X: float array to convert
 *
 *  returns: resulting array after applying natural log to each element of X
 */
void convertArrayToLogProb(int n, float X[n]) {
  for (int i = 0; i < n; i++) {
    X[i] = (float)log(X[i]);
  }
}
