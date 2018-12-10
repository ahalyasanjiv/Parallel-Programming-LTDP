#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int max(int a, int b);
int min(int a, int b);
void print_reverse(char* x, int x_len);
void print_matrix(int m, int n, int A[m][n]);
char* generateSequence(int n);
int get_num_of_elements_in_stage(int i, int x_len, int y_len);

/* Finds the max of two integers */
int max(int a, int b) {
  if (a > b) {
    return a;
  }
  return b;
}

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

// Get number of elements in stage based on stage number and lengths of strings
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
