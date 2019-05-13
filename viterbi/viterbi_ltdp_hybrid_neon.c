#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "hmm_data_gen_hybrid.h"
//#include "arm_neon.h"

int world_rank, world_size;
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
  int S[q],
  int Y[t],
  float I[q],
  float A[q][q],
  float B[q][n]
) {
  double start, end;
  float (*dp1)[q] = malloc(sizeof *dp1 * t); // dp1[i,j] is the prob of most likely path of length i ending in S[j] resulting in the obs sequence
  int (*dp2)[q] = malloc(sizeof *dp2 * t); // dp2[i,j] stores predecessor state of the most likely path of length i ending in S[j] resulting in the obs sequence
  if (world_rank == 0) {
    float min_prob = -1.000;
    float max_prob = -0.001;

    // Initialize dp matrices

    for (int i=0; i<t; i++) {
      for (int j=0; j<q; j++) {
        if (i == 0) {
          int observation = Y[0];
          dp1[0][j] = I[j]+B[j][observation]; // multiple init probability of state S[i] by the prob of observing init obs from state S[i]
          dp2[0][j] = 0;
        }
        else {
          dp1[i][j]=get_rand_float(min_prob,max_prob);
          dp2[i][j]=0;
        }
      }
    }

  }

  int stages_left = t-1;

  // Make excessive processors are not used
  int comm_size = world_size >= stages_left ? stages_left : world_size;
  int ranks[comm_size];
  for (int i = 0; i < comm_size; i++) {
    ranks[i] = i;
  }

  // Get the group of processes in MPI_COMM_WORLD
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);


  // Construct a group containing all of the prime ranks in world_group
  MPI_Group group;
  MPI_Group_incl(world_group, comm_size, ranks, &group);

  // Create a new communicator based on the group
  MPI_Comm comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, group, 0, &comm);
  if (MPI_COMM_NULL == comm) {
  	goto terminate;
  }
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  MPI_Bcast(dp1, t*q, MPI_FLOAT, 0, comm);
  MPI_Bcast(dp2, t*q, MPI_INT, 0, comm);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Barrier(comm);
  int min_stages_per_node = (t-1) / world_size;
  int lp, rp;
  // Assign stages to each node
  if (world_rank != 0) {
    MPI_Recv(&stages_left, 1, MPI_INT, world_rank - 1, 0, comm, MPI_STATUS_IGNORE);
    lp = t - stages_left;
  } else {
    lp = 1;
  }
  if ((t-1) % world_size != 0 && (world_rank + 1) <= (t-1) % world_size) {
    rp = lp + min_stages_per_node + 1;
  } else {
    rp = lp + min_stages_per_node;
  }
  if (stages_left - (rp - lp) <= 0) {
    rp = t;
  }
  if (stages_left == 0) {
    lp = -1; rp = -1;
  }
  stages_left = stages_left - (rp - lp);
  if (world_rank < world_size - 1) {
    MPI_Send(&stages_left, 1, MPI_INT, world_rank + 1, 0, comm);
  }
  printf("WR:%d lp:%d rp:%d \n", world_rank,lp,rp);
  // Forward Phase
  start = MPI_Wtime();
  for (int i = lp; i < rp; i++) {
    #pragma omp parallel for
    for (int j = 0; j < q; j++) {
      float max = -INFINITY;
      int arg_max = -1;
      double curr_log_prob;
      for (int k = 0; k < q; k++) {
        curr_log_prob = dp1[i-1][k] + A[k][j] + B[j][Y[i]];
        // Update max and curr_max if needed
        if (curr_log_prob > max) {
          max = curr_log_prob;
          arg_max = k;
        }
      }
      // Update dp memos
      dp1[i][j] = max;
      dp2[i][j] = arg_max;
    }
  }
  end = MPI_Wtime();
  printf("time: %f\n", (end-start));
  MPI_Barrier(comm);

  // Fixup Phase
  int local_converged = 0;
  int global_converged = 0;
  do {
    // All processes but the first receive a solution vector from the previous process
    if (world_rank > 0) {
      MPI_Recv(dp1[lp-1], q, MPI_FLOAT, world_rank-1, 0, comm, MPI_STATUS_IGNORE);
    }
    // All processes but the last send their last solution vector to the next process
    if(world_rank < world_size - 1) {
      MPI_Send(dp1[rp-1], q, MPI_FLOAT, world_rank + 1, 0, comm);
    }

    if (world_rank == 0) {
      local_converged = 1;
    } else if (!local_converged) {

      #pragma omp parallel for
      for (int i = lp; i < rp; i++) {
        float s1[q]; // holds new soln (corresponding to dp1)
        int s2[q];  // holds new soln (corresponding to dp2)
        // Fix stage i using actual solution to stage i-1
        fix_stage(n,i,q,t,s1,s2,dp1,Y,A,B);
        // If new solution and old solution are parallel, break
        local_converged = is_parallel(t,q,i,s1,dp1) ? 1 : 0;
        if (!local_converged) {
          copy_new_to_old(t,q,i,s1,s2,dp1,dp2);
        }
      }
    }
    MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_LAND, comm);
  } while (global_converged == 0);

  // Backwards Phase
  if (world_rank == 0) {
    // If there are multiple processes, 1st process receives predecessor info from other processes
    if (world_size > 1) {
      int prev_rp = rp;
      for (int i=1; i < world_size; i++) {
        MPI_Recv(&rp, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(dp2[prev_rp], (rp-prev_rp) * q, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
        prev_rp = rp;
      }
    }
    float max = dp1[t-1][0];
    int arg_max = 0;
    float state_prob;
    for (int i=1; i<q; i++) {
      state_prob = dp1[t-1][i];
      if (state_prob > max) {
        max = state_prob;
        arg_max = i;
      }
    }
    // Backtrack and store most probable state sequence in X
    int X[t];
    X[t-1] = S[arg_max];
    if (t > 1) {
      for (int i=t-1; i>0; i--) {
        arg_max = dp2[i][arg_max];
        X[i-1] = S[arg_max];
      }
    }

    print_arr(t,X);
  } else {
    MPI_Send(&rp, 1, MPI_INT, 0, 0, comm);
    MPI_Send(dp2[lp], (rp-lp)*q, MPI_INT, 0, 0, comm);
  }

  terminate:
    MPI_Barrier(MPI_COMM_WORLD);
}

int main() {

  int n,q,t;

  MPI_Init(NULL, NULL);
  // Create initial communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_rank == 0) {
    printf("===========================================================\n"
           "VITERBI LTDP HYBRID PARALLEL ALGORITHM\n");
    printf("Computing the most probable state sequence from a sequence of observations.\n");
    printf("This program will generate the observation sequence and HMM based on the\n"
            "dimensions you specify.\n");

    // printf("Enter the size of the observation space: ");
    // scanf("%d",&n);
    // printf("Enter the size of the state space: ");
    // scanf("%d",&q);
    n = 5;
    q = 5;
    printf("Enter the number of observations in each sequence: ");
    scanf("%d",&t);
  }

  MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&q,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&t,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  int O[n];
  int S[q];
  float I[q];
  float A[q][q];
  float B[q][n];
  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) {
    generate_sequence(q,n,t,O,S,I,A,B);
  }

  MPI_Bcast(O,n,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(S,q,MPI_INT,0,MPI_COMM_WORLD);
  // MPI_Bcast(Y,t,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(I,q,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Bcast(A,q*q,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Bcast(B,q*n,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  // MPI_Status status;
  // MPI_File file_handle;
  // int file_error = MPI_File_open( MPI_COMM_WORLD, "generated_sequence.c", MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
  // if (file_error != MPI_SUCCESS) {
  //   printf("ERROR %d\n", file_error);
  // }
  for( int i = 1; i < world_size; ++i ) {
    MPI_Barrier( MPI_COMM_WORLD );
    if ( world_rank == 0 || i == world_rank ) {
      int start = (t/world_size) * i;
      int end = start + (t/world_size);
      if (start < t) {
        if (end > t) {
          end = t;
        }
        int buffer_size = end - start;
        int buf[buffer_size];
        if (world_rank == 0) {
          char filename[] = "generated_sequence.c";
          FILE * fp = fopen(filename, "r");
          fseek(fp, start, SEEK_SET);
          int count = 0;
          while (!feof (fp) && (count+start) < end) {
            fscanf(fp, "%1d", &buf[count]);
            count++;
          }
          fclose(fp);
          // viterbi(n,q,t,O,S,I,A,B);
          MPI_Send(buf, buffer_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        } else {
          MPI_Recv(buf, buffer_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (int j = 0; j < buffer_size; j++) {
            printf("%d ", buf[j]);
          }
        }

      }
    }
  }
  if (world_rank == 0) {
    int end = (t/world_size);
    char filename[] = "generated_sequence.c";
    int buf[end];
    FILE * fp = fopen(filename, "r");
    fseek(fp, 0, SEEK_SET);
    int count = 0;
    while (!feof (fp) && count < end) {
      fscanf(fp, "%1d", &buf[count]);
      count++;
    }
    for (int j = 0; j < end; j++) {
      printf("%d ", buf[j]);
    }
    fclose(fp);
    // viterbi(n,q,t,O,S,I,A,B);
  }


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
