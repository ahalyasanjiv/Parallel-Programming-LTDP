#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "sw_helpers.h"

struct max_score smith_waterman( char* ref , char* query, int seq, int m, int n);
void backward( char* x, char* y, int m, int n, int score[m][n], int pred[m][n], int max_row, int max_col );

int MATCH = 5;
int MISMATCH = -3;
int SPACE = -4;
int world_rank, world_size;

struct max_score {
  int score;
  int seq;
  int row;
  int col;
};

struct max_score smith_waterman(
  char* ref,
  char* query,
  int seq,
  int m,
  int n
) {
  int num_threads = 2;
  int num_teams = 2;
  int num_stages = get_num_of_stages(m,n);
  int block_size = (int)ceil((float)num_stages / num_teams);

  char ref_alignment[m+n-1];
  char query_alignment[m+n-1];
  struct max_score global_max = {-1,seq,-1,-1};
  struct max_score local_max[num_threads*num_teams];
  for (int i = 0; i<(num_threads*num_teams); i++) {
    local_max[i] = (struct max_score){-1,seq,-1,-1};
  }

  int score[m][n];
  int pred[m][n];

  // Initialize matrices
  // Initialize the first row and first column of matrix to 0
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      score[i][j]=0;
      pred[i][j]=0;
    }
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int team_num = tid/num_teams;
    int lp = team_num * block_size;
    int rp = lp + block_size;

    // Create arrays to store initial read values for up, diagonal, and left
    int up[n];
    int diagonal[n];
    int left[n];
    int start_row, start_col, curr_row, curr_col;

    int num_of_elements_in_stage, elements_to_calculate;
#pragma omp barrier
    for (int i = lp; i < rp; i++)
    {
      start_row = get_start_row(i,m);
      start_col = get_start_col(i,m);
      curr_row = start_row - (tid % num_threads);
      curr_col = start_col + (tid % num_threads);
      num_of_elements_in_stage = get_num_of_elements_in_stage(i,m,n);
      elements_to_calculate = (int)ceil((float)num_of_elements_in_stage/num_threads);
#pragma omp barrier
      for (int j = 0; j < elements_to_calculate; j++)
      {
        if (curr_col < n && curr_row > 0) {
          if (ref[curr_row-1] == query[curr_col-1]) {
            diagonal[curr_col] = score[curr_row-1][curr_col-1] + MATCH;
          } else {
            diagonal[curr_col] = score[curr_row-1][curr_col-1] + MISMATCH;
          }
          up[curr_col] = score[curr_row-1][curr_col] + SPACE;
          left[curr_col] = score[curr_row][curr_col-1] + SPACE;
          curr_col += num_threads;
          curr_row -= num_threads;
        }
      }
      curr_row = start_row - (tid % num_threads);
      curr_col = start_col + (tid % num_threads);
#pragma omp barrier
      for (int j = 0; j < elements_to_calculate; j++) {
        if (curr_col < n && curr_row > 0) {
          calculate_element(up[curr_col],diagonal[curr_col],left[curr_col],&score[curr_row][curr_col], &pred[curr_row][curr_col]);
          if (score[curr_row][curr_col] > local_max[tid].score) {
            local_max[tid].score = score[curr_row][curr_col];
            local_max[tid].row = curr_row;
            local_max[tid].col = curr_col;
          }
          curr_col += num_threads;
          curr_row -= num_threads;
        }
      }
    }
  }

  // Fixup Phase
  bool converged = false;

  int new_score[m][n];
  int new_pred[m][n];

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      new_score[i][j] = 0;
      new_pred[i][j] = 0;
    }
  }

  do {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int team_num = floor((num_teams/num_threads) * (tid/num_threads));
      int lp = block_size;
      int rp = lp + block_size;
      int up[n];
      int diagonal[n];
      int left[n];
      int start_row, start_col, curr_row, curr_col;
      int num_of_elements_in_stage, elements_to_calculate;
      #pragma omp barrier
      for (int i = lp; i < rp; i++)
      {
        num_of_elements_in_stage = get_num_of_elements_in_stage(i,m,n);
        elements_to_calculate = (int)ceil((float)num_of_elements_in_stage/num_threads);
        start_row = get_start_row(i,m);
        start_col = get_start_col(i,m);
        curr_row = start_row - (tid % num_threads);
        curr_col = start_col + (tid % num_threads);

      #pragma omp barrier
        for (int j = 0; j < elements_to_calculate; j++)
        {
          if (curr_col < n && curr_row > 0) {
            if (ref[curr_row-1] == query[curr_col-1]) {
              diagonal[curr_col] = score[curr_row-1][curr_col-1] + MATCH;
            } else {
              diagonal[curr_col] = score[curr_row-1][curr_col-1] + MISMATCH;
            }
            up[curr_col] = score[curr_row-1][curr_col] + SPACE;
            left[curr_col] = score[curr_row][curr_col-1] + SPACE;
            curr_col += num_threads;
            curr_row -= num_threads;
          }
        }
        curr_row = start_row - tid;
        curr_col = start_col + tid;
        #pragma omp barrier
        for (int j = 0; j < elements_to_calculate; j++) {
          if (curr_col < n && curr_row > 0) {
            calculate_element(up[curr_col],diagonal[curr_col],left[curr_col],&new_score[curr_row][curr_col],&new_pred[curr_row][curr_col]);
            if (new_score[curr_row][curr_col] > local_max[tid].score) {
              local_max[tid].score = new_score[curr_row][curr_col];
              local_max[tid].row = curr_row;
              local_max[tid].col = curr_col;
            }
            curr_col += num_threads;
            curr_row -= num_threads;
          }
        }
        #pragma omp barrier
        if (is_parallel(m,n,start_row,start_col,new_score,score)) {
          converged = true;
          break;
        }
        copy_new_soln(m,n,start_row,start_col,new_score,new_pred,score,pred);
      }
    }
  } while (!converged);

  for (int i = 0; i < num_threads * num_teams; i++) {
    if (local_max[i].score > global_max.score) {
      global_max = local_max[i];
    }
  }
  return global_max;
}

void backward(
  char* x,
  char* y,
  int m,
  int n,
  int score[m][n],
  int pred[m][n],
  int max_row,
  int max_col
) {
  int alignment_len = 0;
  char result_x_alignment[m+n-1];
  char result_y_alignment[m+n-1];
  int i = max_row;
  int j = max_col;
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

int main(int argc, char* argv[]) {

  srand(time(NULL));

  // Create a new MPI derived datatype for max score
  MPI_Datatype mpi_max_score;

  // Create communicator
  MPI_Init(NULL, NULL);

  MPI_Type_contiguous(4,MPI_INT,&mpi_max_score);
  MPI_Type_commit(&mpi_max_score);

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int ref_len,query_len,ref_seq_len;
  char* ref;
  char* query;

  if (world_rank == 0) {
    printf("===========================================================\n"
           "SMITH WATERMAN PARALLEL ALGORITHM\n");
    printf("Computing local alignment.\n");
    printf("===========================================================\n");
    printf("Enter the number of sequences in the reference: ");
    scanf("%d",&ref_len);
    printf("Enter the string size for each reference sequence: ");
    scanf("%d",&ref_seq_len);
    printf("Enter the string size for the query string: ");
    scanf("%d",&query_len);

    // Generate a new reference sequence
    query = generate_query_sequence(query_len);
    printf("Query string: %s\n",query);
  } else {
    ref = malloc(ref_seq_len);
    query = malloc(query_len);
  }
  MPI_Bcast(&ref_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ref_seq_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&query_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(query, query_len, MPI_CHAR, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  int m = ref_seq_len+1;
  int n = query_len+1;

  // Split sequences between slave nodes
  int seqs_per_comm = (ref_len + (world_size - 1) - 1) / (world_size - 1);

  struct max_score global_max = {-1,-1,-1,-1};

  // Master node will send sequences to appropriate slave node
  int slave_idx = 1;
  double start, end;
  start = MPI_Wtime();
  if (world_rank == 0) {
    int seq_idx = 0;
    ref = generate_ref_sequence(ref_len,ref_seq_len);
    for (int i = 0; i < ref_len; i++) {
      for (int j = 0; j < ref_seq_len; j++) {
        printf("%c", ref[i*ref_seq_len + j]);
      }
      printf(" ");
      MPI_Send(ref + (i * ref_seq_len), ref_seq_len, MPI_CHAR, slave_idx, 0, MPI_COMM_WORLD);
      seq_idx++;
      if (seq_idx >= seqs_per_comm && (i + 1) < ref_len ) {
        seq_idx = 0;
        slave_idx++;
      }
    }
    printf("\n");
  } else {

    // Assign each communicator a chunk of the database
    int init_seq = (world_rank - 1) * seqs_per_comm;
    int final_seq = init_seq + seqs_per_comm;

    int score[m][n];
    int pred[m][n];

    if (init_seq > ref_len - 1){
      goto terminate;
    }

    if (final_seq > ref_len) {
      final_seq = ref_len;
    }

    for (int k = init_seq; k < final_seq; k++) {
      int num_of_stages = m+n-3;
      int start_i, start_j, curr_i, curr_j;
      int num_of_elements_in_stage;
      MPI_Recv(ref, ref_seq_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      struct max_score temp_max = smith_waterman(ref + ((k-init_seq) * ref_seq_len), query, k, m, n);
      if (temp_max.score > global_max.score) {
        global_max = temp_max;
        global_max.seq = k;
      }
    }
  }
  end = MPI_Wtime();
  printf("time: %f\n", end-start);


  // Determine global maximum across

  if (world_rank == 0) {
    struct max_score local_max = {-1,-1,-1,-1};
    for (int i=1; i <= slave_idx; i++) {
      MPI_Recv(&local_max, 1, mpi_max_score, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (local_max.score > global_max.score) {
      global_max = local_max;
    }
  } else {
    MPI_Send(&global_max, 1, mpi_max_score, 0, 0, MPI_COMM_WORLD);
  }

  // Report the best local alignment
  if (world_rank == 0) {
    printf("Max Score: %d\n", global_max.score);
    printf("Sequence: %d\n", global_max.seq);
  }

  terminate:
    MPI_Barrier(MPI_COMM_WORLD);
    free(ref);
    free(query);
    MPI_Finalize();
  return 0;
}
