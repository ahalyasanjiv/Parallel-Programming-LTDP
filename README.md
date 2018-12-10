# Parallel DP Algorithms
Parallel and serial implementations of Viterbi and Smith-Waterman using OpenMP
(inspired by Parallelizing Dynamic Programming through Rank Convergence by Maleki et al.)

## Requirements
The programs are all written in C.
### Compiler
To compile these programs, I used GCC Version 8.
You can find information on installing GCC [here](https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html).
### OpenMP
The GCC/G++ compilers can run programs with OpenMP directives as is.

## Smith-Waterman
Smith-Waterman computes the optimal local alignment for a reference string and
query string. The programs will dynamically generate randomized DNA sequences
to use as the reference and query based on the user input. The user provides
the length for the reference and the length for the query (these values may be
different).

### To Run
First navigate to the smith-waterman directory.
```
cd smith-waterman/
```

#### Serial Version
This version computes local alignment by computing values in the score matrix row by row,
iterating through each element in a row before moving onto the next.

Run the following commands:
```
gcc-8 -fopenmp smith_waterman_serial.c -o sw
./sw
```

#### Parallel Version
This version computes local alignment by computing values stage by stage, in which each
stage is an anti-diagonal in the matrix. The stages themselves are computed in order,
but the elements within the stage are computed in parallel.

Run the following commands:
```
gcc-8 -fopenmp smith_waterman_par.c -o sw
./sw
```
#### LTDP Parallel Version
This version computes local alignment by computing values stage by stage, in which each
stage is an anti-diagonal in the matrix. Both the stages and the elements within the
stage are computed in parallel (fine-grained and coarse-grained parallelism).

Run the following commands:
```
gcc-8 -fopenmp smith_waterman_ltdp.c -o sw
./sw
```
