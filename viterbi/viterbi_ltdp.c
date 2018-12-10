#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "hmm_data_gen.h"
#include "viterbi_helpers.h"

void viterbi(int n, int q, int t, int O[n], int S[q],float I[q], int Y[t], float A[q][q], float B[q][n]);
void fix_stage(int n, int lp, int q, int t, float s1[q], int s2[q], float dp1[q][t], int Y[t], float A[q][q], float B[q][n]);
void copy_new_stage_to_old(int q, int t, int stage_num, float s1[q], int s2[q], float dp1[q][t], int dp2[q][t]);
bool is_parallel(int q, int t, int comp_stage_idx, float s[q], float dp1[q][t]);

/*
 * Function: viterbi
 * --------------------
 *  Computes the most likely hidden state sequence based on a sequence of observations using Viterbi Algorithm
 *
 *  n: number of possible observations
 *  q: number of possible states
 *  t: length of observed sequence
 *  O: observation space
 *  S: state space
 *  I: prior probability - I[i] is the prior probability of S[i]
 *  Y: sequence of observations - Y[t] = i if observation at time t is O[i]
 *  A: transition probability - A[i,j] is the probability of going from state S[i] to S[j]
 *  B: emission probability - B[i,j] is the probability of observing O[j] given state S[i]
 *
 *  returns: the most likely hidden state sequence corresponding to given observations Y
 */
void viterbi(
  int n,
  int q,
  int t,
  int O[n],
  int S[k],
  float I[k],
  int Y[t],
  float A[k][k],
  float B[k][n]
) {
  int p = 4;
  float dp1[q][t]; // dp1[i,j] is the prob of most likely path of length j ending in S[i] resulting in the obs sequence
  int dp2[q][t]; // dp2[i,j] stores predecessor state of the most likely path of length j ending in S[i] resulting in the obs sequence
  int segment_size = t / p;

  float min_prob = -1.000;
  float max_prob = -0.001;

  /* Initialize dp matrices */
  for (int i=0; i<q; i++) {
    for (int j=0; j<t; j++) {
      if (j==0) {
        int observation = Y[0];
        dp1[i][0] = I[i]+B[i][observation]; // multiple init probability of state S[i] by the prob of observing init obs from state S[i]
        dp2[i][0] = 0;
      }
      else {
        dp1[i][j]=get_rand_float(min_prob,max_prob);
        dp2[i][j]=0;
      }
    }
  }
  int max_count = p;
  int threads_present = 0;
  // Forward Phase
  #pragma omp parallel
  {
  omp_set_nested(1);
  #pragma omp for schedule(static)
  for (int i=0; i<p; i++) {
    int lp = segment_size * i;
    if (i == 0) {
      lp = 1;
    }
    int rp = segment_size * (i+1) - 1;
    if (i == p - 1) {
      rp = t - 1;
    }
    // for each stage that processor i is responsible for
    double t1, t2;
    t1 = omp_get_wtime();
    for (int j = lp; j <= rp; j++) {
      for (int k = 0; k < q; k++) {
        float max = -INFINITY;
        int arg_max = -1;
        double curr_log_prob, sub_prob;
        for (int l = 0; l < q; l++) {
          sub_prob = dp1[l][j-1];
          threads_present++;
          while (threads_present < max_count) {}
          threads_present--;
          curr_log_prob = sub_prob + A[l][k] + B[k][Y[j]];
          // // Update max and curr_max if needed
          if (curr_log_prob > max) {
            max = curr_log_prob;
            arg_max = l;
          }
        }
        // Update dp memos
        dp1[k][j] = max;
        dp2[k][j] = arg_max;
      }
      if (j == rp) {
        max_count -= 1;
      }
    }
    printf("time: %f\n", omp_get_wtime()-t1);
  }
  }

  // Fixup Phase
  bool converged = false;
  bool conv[p]; // conv[i] denotes whether segment i has converged
  for (int i=0; i<p; i++) {
    conv[i] = false;
  }
  int count = 0;
  do {
    count++;
    #pragma omp parallel for
    for (int i = 1; i < p; i++) {
      int lp = segment_size * i;
      int rp = lp + segment_size - 1;
      if (i == p - 1) {
        rp = t - 1;
      }
      bool parallel = false;
      float s1[q]; // holds new soln (corresponding to dp1)
      int s2[q];  // holds new soln (corresponding to dp2)
      for (int j=lp; j<=rp; j++) {
        // Fix stage j using actual solution to stage j-1
        fixStage(n,j,q,t,s1,s2,dp1,Y,A,B);
        // If new solution and old solution are parallel, break
        parallel = is_parallel(q,t,j,s1,dp1);
        if (parallel) {
          conv[i] = true;
          break;
        }
        copy_new_stage_to_old(q,t,j,s1,s2,dp1,dp2);
      }
      bool local_conv = true;
    }
    converged = true;
    for (int i=1; i<p; i++) {
      if (!conv[i]) {
        converged = false;
        break;
      }
    }
  } while (!converged);


  // Get last state in path
  float max = dp1[0][t-1];
  int arg_max = 0;
  float state_prob;
  for (int i=1; i<q; i++) {
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

}

/* Copy new stage to old */
void copy_new_stage_to_old(int q, int t, int stage_num, float s1[q], int s2[q], float dp1[q][t], int dp2[q][t]) {
  for (int i = 0; i < q; i++) {
    dp1[i][stage_num] = s1[i];
    dp2[i][stage_num] = s2[i];
  }
}

/*
 * Function: fix_stage
 * --------------------
 *  Calculates new solution vector of values for fixup phase.
 *
 *  n: number of observations
 *  lp: the stage index to be fixed
 *  q: number of possible states
 *  t: length of observed sequence
 *  s1: new probability solution vector
 *  s2: new predecessor solution vector
 *  dp1: matrix containing probabilties calculated in forward phase
 *  Y: sequence of observations - Y[t] = i if observation at time t is O[i]
 *  A: transition probability - A[i,j] is the probability of going from state S[i] to S[j]
 *  B: emission probability - B[i,j] is the probability of observing O[j] given state S[i]
 */
void fix_stage(
  int n,
  int lp,
  int q,
  int t,
  float s1[q],
  int s2[q],
  float dp1[q][t],
  int Y[t],
  float A[q][q],
  float B[q][n]
) {
  for (int i = 0; i < q; i++) {
    float max = -INFINITY;
    int arg_max = -1;
    double curr_log_prob;
    for (int j = 0; j < q; j++) {
      curr_log_prob = dp1[j][lp-1] + A[j][i] + B[i][Y[lp]];
      // Update max and curr_max if needed
      if (curr_log_prob > max) {
        max = curr_log_prob;
        arg_max = j;
      }
    }
    s1[i] = max;
    s2[i] = arg_max;
  }
}


/*
 * Function: is_parallel
 * --------------------
 *  Checks if a new solution vector is parallel to the corresponding column
 *  vector in the dp solution matrix.
 *
 *  q: number of possible states
 *  t: length of observed sequence
 *  comp_stage_idx: index of stage being computed
 *  s: new solution vector
 *  dp: matrix containing old solution vector
 *
 *  returns: true if s and corresponding column in dp are parallel
 */
bool is_parallel(int q, int t, int comp_stage_idx, float s[q], float dp[q][t]) {
  for (int i = 1; i < q; i++) {
    if ((s[i-1]-dp[i-1][comp_stage_idx]) != (s[i]-dp[i][comp_stage_idx])) {
      return false;
    }
  }
  return true;
}

int main() {
  // O[0:happy, 1:grumpy]
  // S[0:sunny, 1:rainy]
  // const int t = 6;
  // int n = 2;
  // int q = 2;
  // int t = 8;
  // int O[] = {0,1};
  // int S[] = {0,1};
  // float I[2] = {log(0.67), log(0.33)};
  // float A[2][2] = {{0.8,0.2},{0.4,0.6}};
  // float B[2][2] = {{0.8,0.2},{0.4,0.6}};
  // convert_to_log_prob(2,2,A);
  // convert_to_log_prob(2,2,B);
  // int Y[8] = {0,0,1,1,1,0,1,0};
  int n = 100;
  int q = 100;
  int t = 100;
  int O[n];
  int S[q];
  int Y[t];
  float I[q];
  float A[q][q];
  float B[q][n];
  convert_array_to_log_prob(2,I);
  convert_to_log_prob(2,2,A);
  convert_to_log_prob(2,2,B);
  generate_sequence(q,n,t,O,S,Y,I,A,B);

  // int O[n];
  // int S[q];
  // float I[n];
  // float A[q][q];
  // float B[q][n];
  // int Y[t];
  //
  // for (int i = 0; i < q; i++) {
  //   float max_prob = 1.0;
  //   for (int j = 0; j < q; j++) {
  //     A[i][j] = get_rand_float(0.0,max_prob);
  //     max_prob = max_prob - A[i][j];
  //   }
  // }
  //
  // for (int i = 0; i < q; i++) {
  //   float max_prob = 1.0;
  //   for (int j = 0; j < n; j++) {
  //     B[i][j] = get_rand_float(0.0,max_prob);
  //     max_prob = max_prob - B[i][j];
  //   }
  // }

  viterbi(n,q,t,O,S,I,Y,A,B);
  return 0;
}
