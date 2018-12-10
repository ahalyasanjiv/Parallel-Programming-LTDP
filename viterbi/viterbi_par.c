#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "viterbi_helpers.h"

void viterbi( int n, int k, int t, int O[n], int S[k], float I[k], int Y[t], float A[k][k], float B[k][n]);

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
        dp1[i][0] = I[i]*B[i][observation]; // multiple init probability of state S[i] by the prob of observing init obs from state S[i]
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
  for (int i=1; i<t; i++) {
    #pragma omp parallel
    {
    double start;
    double end;
    start = omp_get_wtime();
    #pragma omp for

      // First inner loop is iterating through possible states //shared(dp1, dp2, A, B, Y, i, k)
      for (int j=0; j<k; j++) {
        float max = -1.0;
        int arg_max = -1;
        double curr_prob;
        // Second inner loop is iterating though possible states that could have preceded state S[j]
        for (int q=0; q<k; q++) {
          // dp1[q,i-1] - prob of most likely path from prev stage ending in S[q]
          // A[q,j] - prob of going from state S[q] to S[j]
          // B[j,Y[i]] - prob of observing O[Y[i]] state S[j]
          curr_prob = dp1[q][i-1] * A[q][j] * B[j][Y[i]];
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
      end = omp_get_wtime();
      printf("Work took %f sec. time.\n", end-start);
    }
  }

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

  print_arr(6,X);
}

int main() {
  // O[0:happy, 1:grumpy]
  // S[0:sunny, 1:rainy]
  // const int t = 6;
  int O[] = {0,1};
  int S[] = {0,1};
  float I[2] = {0.67, 0.33};
  float A[2][2] = {{0.8,0.2},{0.4,0.6}};
  float B[2][2] = {{0.8,0.2},{0.4,0.6}};
  int Y[6] = {0,0,1,1,1,0};
  viterbi(2,2,6,O,S,I,Y,A,B);
  return 0;
}
