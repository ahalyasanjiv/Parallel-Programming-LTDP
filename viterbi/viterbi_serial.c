#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Function Prototypes */
int max(int a, int b);
void viterbi( int n, int k, int t, int O[n], int S[k], float I[k], int Y[t], float A[k][k], float B[k][n]);
void print_arr(int t, int X[t]);
void print_fl_matrix(int m, int n, float A[m][n]);
void print_matrix(int m, int n, int A[m][n]);

/* Finds the max of two integers */
int max(int a, int b) {
  if (a > b) {
    return a;
  }
  return b;
}

/* Returns the most likely hidden state sequence corresponding to given observations Y */
void viterbi(
  int n, // number of possible observations
  int k, // number of possible states
  int t, // length of observed sequence
  int O[n], // observation space
  int S[k], // state space
  float I[k], // I[i] is the initial probability of S[i]
  int Y[t], // sequence of observations - Y[t] = i if observation at time t is O[i]
  float A[k][k], // A[i,j] is the transition probability of going from state S[i] to S[j]
  float B[k][n] // B[i,j] is the probability of observing O[j] given state S[i]
) {
  // Initialize DP matrices
  float dp1[k][t]; // dp1[i,j] is the prob of most likely path of length j ending in S[i] resulting in the obs sequence
  int dp2[k][t]; // dp2[i,j] stores predecessor state of the most likely path of length j ending in S[i] resulting in the obs sequence
  for (int i=0; i<k; i++) {
    for (int j=0; j<t; j++) {
      if (j==0) {
        int observation = Y[0];
        dp1[i][0] = I[i]+B[i][observation]; // multiple init probability of state S[i] by the prob of observing init obs from state S[i]
        dp2[i][0] = 0;
      }
      else {
        dp1[i][j]=0.0;
        dp2[i][j]=0;
      }
    }
  }
  // Forward algorithm
  // Outer loop is iterating through t time stages
  clock_t time_taken;
  time_taken = clock();
  for (int i=1; i<t; i++) {
    // First inner loop is iterating through possible states
    for (int j=0; j<k; j++) {
      float max = -INFINITY;
      int arg_max = -1;
      double curr_prob;
      // Second inner loop is iterating though possible states that could have preceded state S[j]
      for (int q=0; q<k; q++) {
        // dp1[q,i-1] - prob of most likely path from prev stage ending in S[q]
        // A[q,j] - prob of going from state S[q] to S[j]
        // B[j,Y[i]] - prob of observing O[Y[i]] state S[j]
        curr_prob = dp1[q][i-1] + A[q][j] + B[j][Y[i]];
        // Update max and curr_max if needed
        if (curr_prob > max) {
          max = curr_prob;
          arg_max = q;
        }
      }
      // Update dp memos
      dp1[j][i] = max;
      dp2[j][i] = arg_max;
    }
  }
  time_taken = clock() - time_taken;
  printf("Time: %f\n", (double)time_taken/CLOCKS_PER_SEC);
  // Get last state in path
  float max = dp1[0][t-1];
  int arg_max = 0;
  float state_prob;
  for (int i=1; i<k; i++) {
    state_prob = dp1[i][t-1];
    if (state_prob > max) {
      max = state_prob;
      arg_max = i;
    }
  }
  int X[t];
  X[t-1] = S[arg_max];
  // Backward algorithm
  for (int i=t-1; i>0; i--) {
    arg_max = dp2[arg_max][i];
    X[i-1] = S[arg_max];
  }

  print_arr(8,X);
  print_fl_matrix(k,t,dp1);
  print_matrix(k,t,dp2);
  return;
}

void print_arr(int t, int X[t]) {
  for (int i=0; i<t; i++) {
    printf("%d ", X[i]);
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

int main() {
  // O[0:happy, 1:grumpy]
  // S[0:sunny, 1:rainy]
  // const int t = 6;
  int O[] = {0,1};
  int S[] = {0,1};
  float I[2] = {log(0.67), log(0.33)};
  float A[2][2] = {{log(0.8),log(0.2)},{log(0.4),log(0.6)}};
  float B[2][2] = {{log(0.8),log(0.2)},{log(0.4),log(0.6)}};
  int Y[8] = {0,0,1,1,1,0,1,0};
  viterbi(2,2,8,O,S,I,Y,A,B);
  return 0;
}
