/**
 * syrk.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// polybenchUtilFuncts.h
// Scott Grauer-Gray (sgrauerg@gmail.com)
// Functions used across hmpp codes

#ifndef POLYBENCH_UTIL_FUNCTS_H
#define POLYBENCH_UTIL_FUNCTS_H

// define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a) {
  if (a < 0) {
    return (a * -1);
  } else {
    return a;
  }
}

float percentDiff(double val1, double val2) {
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)) {
    return 0.0f;
  }

  else {
    return 100.0f *
           (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
  }
}

#endif // POLYBENCH_UTIL_FUNCTS_H

int main();

// define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1

/* Problem size */
#define N 1024
#define M 1024

/* Declared constant values for alpha and beta */
/* (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *A, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      A[i * M + j] = ((DATA_TYPE)i * j) / N;
    }
    for (j = 0; j < M; j++) {
      C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
      D[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
    }
  }
}

void compareResults(DATA_TYPE *C, DATA_TYPE *D) {
  int i, j, fail;
  fail = 0;

  // Compare C with D
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      if (percentDiff(C[i * M + j], D[i * M + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

void syrk(DATA_TYPE *A, DATA_TYPE *C) {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      C[i * M + j] *= beta;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
      }
    }
  }
}

void GPU__syrk(DATA_TYPE *A, DATA_TYPE *D) {
  int i, j;
  double t_start, t_end;

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (D + 1048575))
  || ((void*) (D + 0) > (void*) (A + 1048575)));
  #pragma omp target data map(to: A[0:1048576]) map(tofrom: D[0:1048576]) if(!RST_AI1)
  {
#pragma omp target teams distribute parallel for if(!RST_AI1)
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      D[i * M + j] *= beta;
    }
  }

#pragma omp target teams distribute parallel for if(!RST_AI1)
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      int k;
      for (k = 0; k < M; k++) {
        D[i * M + j] += alpha * A[i * M + k] * A[j * M + k];
      }
    }
  }
}

  return;
}

inline void update(DATA_TYPE *D) { DATA_TYPE dd = D[0]; }

int main() {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *C;
  DATA_TYPE *D;

  // A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  // C = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  // D = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  cudaMallocHost((void**)&A, N*M*sizeof(DATA_TYPE));
  cudaMallocHost((void**)&C, N*M*sizeof(DATA_TYPE));
  cudaMallocHost((void**)&D, N*M*sizeof(DATA_TYPE));

  printf("<<DawnCC Symmetric rank-k operations >>\n");
  printf("N: %d\n", N);
  printf("M: %d\n", M);

  init_arrays(A, C, D);

  t_start = rtclock();
  GPU__syrk(A, D);

  update(D);

  t_end = rtclock();
  printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

  // t_start = rtclock();
  // syrk(A, C);
  // t_end = rtclock();
  // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  // compareResults(C, D);

  // free(A);
  // free(C);
  // free(D);
  cudaFreeHost(A);
  cudaFreeHost(C);
  cudaFreeHost(D);
  return 0;
}

