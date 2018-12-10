# Parallel DP Algorithms
Parallel and serial implementations of Viterbi and Smith-Waterman using OpenMP
(inspired by [Parallelizing Dynamic Programming through Rank Convergence](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ppopp163-maleki.pdf) by Maleki et al.)


## Requirements
The programs are all written in C.
### Compiler
To compile these programs, I used GCC Version 8.
You can find information on installing GCC [here](https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html).
### OpenMP
The GCC/G++ compilers can run programs with OpenMP directives as is.


## Documentation
All functions are documented in their respective files with descriptions on their
purpose, their parameters, and what they return.
I have commented within the functions as well to describe their implementations.


## Smith-Waterman
Smith-Waterman computes the optimal local alignment for a reference string and
query string. The programs will dynamically generate randomized DNA sequences
to use as the reference and query based on the user input. The user provides
the length for the reference and the length for the query (these values may be
different).
This is different from Needleman-Wunsch, which finds the global optimal alignment.

### Auxiliary File: smith_waterman_helpers.h
This file contains helper functions that are shared amongst the three
different implementations of Smith-Waterman.

### To Run
First navigate to the smith-waterman directory.
```
cd smith-waterman/
```

#### Serial Version: smith_waterman_serial.c
This version computes local alignment by computing values in the score matrix row by row,
iterating through each element in a row before moving onto the next.

Run the following commands:
```
gcc-8 -fopenmp smith_waterman_serial.c -o sw
./sw
```

#### Parallel Version: smith_waterman_par.c
This version computes local alignment by computing values stage by stage, in which each
stage is an anti-diagonal in the matrix. The stages themselves are computed in order,
but the elements within the stage are computed in parallel.

Run the following commands:
```
gcc-8 -fopenmp smith_waterman_par.c -o sw
./sw
```
#### LTDP Parallel Version: smith_waterman_ltdp.c
This version computes local alignment by computing values stage by stage, in which each
stage is an anti-diagonal in the matrix. Both the stages and the elements within the
stage are computed in parallel (fine-grained and coarse-grained parallelism). This
implementation is based on the rank convergence property.

Run the following commands:
```
gcc-8 -fopenmp smith_waterman_ltdp.c -o sw
./sw
```


## Viterbi
The Viterbi algorithm is a dynamic programming algorithm that is used to derive
the most probable sequence of hidden states from a sequence of observations. The
statistical model of hidden states, observations, and the probability distributions
that show how these states and observations are related is known as a Hidden Markov Model.
NOTE: Increasing the size of the observation space and state space can greatly affect
runtime for Viterbi.

### Data Generator: hmm_data_gen.h
This file will be used in all three implementations to randomly generate the observation
sequence to be used as input for Viterbi. The user will specify the size of the
observation space, size of the state space, and length of the sequence of observations.

### Auxiliary File: viterbi_helpers.h
This file contains helper functions that are shared amongst the three
different implementations of Viterbi.

### To Run
First navigate to the viterbi directory.
```
cd viterbi/
```

#### Serial Version: viterbi_serial.c
This version computes the most probable sequence serially column by column, with
each element in each column being computed in order.

Run the following commands:
```
gcc-8 -fopenmp viterbi_serial.c -o viterbi
./viterbi
```

#### Parallel Version: viterbi_par.c
This version computes the most probable sequence column by column, with
elements within a column being computed in parallel.

Run the following commands:
```
gcc-8 -fopenmp viterbi_par.c -o viterbi
./viterbi
```
#### LTDP Parallel Version: viterbi_ltdp.c
This version computes the most probable sequence with stages (in this
case, columns) being computed in parallel. Each processor will be responsible
for computing a sequence of columns, and each processor will be running in parallel.
For example, if there are 8 stages and 4 processors, the first processor will compute
stages 1-2, the second will compute 3-4, etc.

Run the following commands:
```
gcc-8 -fopenmp viterbi_ltdp.c -o viterbi
./viterbi
```
