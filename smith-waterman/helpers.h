#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int max(int a, int b);
void print_reverse(char* x, int x_len);
void print_matrix(int m, int n, int A[m][n]);
char* generateSequence(int n);

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
