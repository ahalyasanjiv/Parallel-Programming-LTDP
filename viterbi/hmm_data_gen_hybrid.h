#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "viterbi_hybrid_helpers.h"

void generate_array(int n, float M[n]);
int get_element_idx(int n, float M[n], float prob);
void generate_sequence(int num_hidden_states, int num_observations, int seq_len, int O[num_observations], int S[num_hidden_states], float I[num_hidden_states], float A[num_hidden_states][num_hidden_states],float B[num_hidden_states][num_observations]);

#define ROUNDF(f, c) (((float)((int)((f) * (c))) / (c)))

/* Generates probability matrix */
void generate_matrix(int m, int n, float M[m][n]) {
  float min_prob = 0.000000;
  float max_prob = 1.000000;
  float rand_prob;
  for (int i = 0; i < m; i++) {
    float sum = 0.000000;
    for (int j = 0; j < n; j++) {
      rand_prob = ROUNDF(min_prob + (float)rand()/RAND_MAX * (max_prob - min_prob),1000000);
      sum += rand_prob;
      M[i][j] = rand_prob;
    }
    for (int j = 0; j < n; j++) {
      M[i][j] /= sum;
      M[i][j] = log(M[i][j]);
    }
  }
}

/* Generates probability array */
void generate_array(int n, float M[n]) {
  float min_prob = 0.000000;
  float max_prob = 1.000000;
  float sum = 0.000000;
  float rand_prob;
  for (int i = 0; i < n; i++) {
    rand_prob = ROUNDF(min_prob + (float)rand()/RAND_MAX * (max_prob - min_prob),1000000);
    sum += rand_prob;
    M[i] = rand_prob;
  }
  for (int i = 0; i < n; i++) {
    M[i] /= sum;
    M[i] = log(M[i]);
  }
}

/* Gets element index based on a random probability and an incrementally summed probability array */
int get_element_idx(int n, float M[n], float prob) {
  int i;
  for (i = 0; i < n-1; i++){
    if (M[i] >= prob) {
      return i;
    }
  }
  return i;
}

void generate_sequence(
  int num_hidden_states,
  int num_observations,
  int seq_len,
  int O[num_observations],
  int S[num_hidden_states],
  float I[num_hidden_states],
  float A[num_hidden_states][num_hidden_states],
  float B[num_hidden_states][num_observations]
) {

  float I_sum_array[num_hidden_states];
  float A_sum_matrix[num_hidden_states][num_hidden_states];
  float B_sum_matrix[num_hidden_states][num_observations];

  srand(time(NULL));
  generate_matrix(num_hidden_states,num_hidden_states,A);
  generate_matrix(num_hidden_states,num_observations,B);
  generate_array(num_hidden_states,I);
  int states_seq[seq_len]; // state sequence array
  for (int i = 0; i < num_observations; i++) {
    O[i] = i;
  }
  for (int i = 0; i < num_hidden_states; i++) {
    S[i] = i;
  }
  // Initiate the initial element of I_sum_array
  I_sum_array[0] = exp(I[0]);
  // Initiate the initial column of A_sum_array and B_sum_matrix
  for (int i = 0; i < num_hidden_states; i++) {
    A_sum_matrix[i][0] = exp(A[i][0]);
    B_sum_matrix[i][0] = exp(B[i][0]);
  }
  // Generate rest of elements in I_sum_array, A_sum_matrix, and B_sum_matrix
  for (int i = 1; i < num_hidden_states; i++) {
    I_sum_array[i] = I_sum_array[i-1] + exp(I[i]);
  }

  for (int i = 0; i < num_hidden_states; i++) {
    for (int j = 1; j < num_hidden_states; j++) {
      A_sum_matrix[i][j] = A_sum_matrix[i][j-1] + exp(A[i][j]);
    }
    for (int j = 1; j < num_observations; j++) {
      B_sum_matrix[i][j] = B_sum_matrix[i][j-1] + exp(B[i][j]);
    }
  }

  // Generate state sequence states
  float min_prob = 0.000000;
  float max_prob = 1.000000;
  float rand_prob = ROUNDF(min_prob + (float)rand()/RAND_MAX * (max_prob - min_prob),1000000);
  int prev_state;
  states_seq[0] = get_element_idx(num_hidden_states,I,rand_prob);
  for (int i = 1; i < seq_len; i++) {
    prev_state = states_seq[i-1];
    rand_prob = ROUNDF(min_prob + (float)rand()/RAND_MAX * (max_prob - min_prob),1000000);
    states_seq[i] = get_element_idx(num_hidden_states,A_sum_matrix[prev_state],rand_prob);
  }

  // Generate observation sequence Y
  int observation_idx;
  int state_idx;
  FILE * fp;
  int i;

  /* open the file for writing*/
  fp = fopen ("generated_sequence.c","w");
  for (int i = 0; i < seq_len; i++) {
    state_idx = states_seq[i];
    rand_prob = ROUNDF(min_prob + (float)rand()/RAND_MAX * (max_prob - min_prob),1000000);
    fprintf (fp, "%d", get_element_idx(num_observations,B_sum_matrix[state_idx],rand_prob));
  }


   /* close the file*/
   fclose (fp);
  // for (int i = 0; i < num_hidden_states; i++) {
  //   printf("%d \n", S[i]);
  // }
  // FILE * fPtr;
  // fPtr = fopen("data/generated_sequence.txt", "w");
  // if (fPtr == NULL) {
  //   printf("Unable to create file.");
  //   exit(EXIT_FAILURE);
  // }
  // for (int i = 0; i < num_observations; i++) {
  //   fprintf(fPtr, "%d ", O[i]);
  // }
  // fputs("\n", fPtr);
  // for (int i = 0; i < num_hidden_states; i++) {
  //   fprintf(fPtr, "%d ", S[i]);
  // }
  // fputs("\n", fPtr);
  // for (int i = 0; i < num_hidden_states; i++) {
  //   fprintf(fPtr, "%d ", S[i]);
  // }
  // fputs("\n", fPtr);
  // for (int i = 0; i < num_hidden_states; i++) {
  //   fputs(Y[i], fPtr);
  // }
  // fputs("\n", fPtr);
  // for (int i = 0; i < num_hidden_states; i++) {
  //   for (int j = 0; j < num_hidden_states; j++) {
  //     fputs(A[i][j], fPtr);
  //   }
  // }
  // fputs("\n", fPtr);
  // for (int i = 0; i < num_hidden_states; i++) {
  //   for (int j = 0; j < num_observations; j++) {
  //     printf("%f",B[i][j]);
  //   }
  //   printf("\n");
  // }
  // fclose(fPtr);
}
