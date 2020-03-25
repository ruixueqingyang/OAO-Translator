/**
 * gemm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <omp.h>
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
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6); // sec秒 usec微秒
}

//绝对值
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 3072
#define NJ 3072
#define NK 3072

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#define GPU_DEVICE 1

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}

void GPU__gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 9437183))
  || ((void*) (B + 0) > (void*) (A + 9437183)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (C + 9437183))
  || ((void*) (C + 0) > (void*) (A + 9437183)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (C + 9437183))
  || ((void*) (C + 0) > (void*) (B + 9437183)));
  #pragma omp target data map(to: A[0:9437184],B[0:9437184]) map(tofrom: C[0:9437184]) if(!RST_AI1)
  {
  #pragma omp target teams distribute parallel for collapse(2) if(!RST_AI1)
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
  }
  return;
}

//初始化，2048*2048大小
void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *C_OMP) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
      C_OMP[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  }
}

void compareResults(DATA_TYPE *C, DATA_TYPE *C_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  // Compare C1 and C2
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      if (percentDiff(C[i * NJ + j], C_outputFromGpu[i * NJ + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) //匹配
      {
        fail++;
      }
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

inline void update(DATA_TYPE *C_outputFromGpu) {
  DATA_TYPE cc = C_outputFromGpu[0];
}

// int main(int argc, char *argv[])
int main() {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *C_outputFromGpu;

  const unsigned long long int threshold = (unsigned long long int)128 * 1024;
  if(threshold > NI*NK*sizeof(DATA_TYPE)){
    A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
  }else{
    cudaMallocHost((void**)&A, NI*NK*sizeof(DATA_TYPE));
  }

  if(threshold > NK*NJ*sizeof(DATA_TYPE)){
    B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
  }else{
    cudaMallocHost((void**)&B, NK*NJ*sizeof(DATA_TYPE));
  }

  if(threshold > NI*NJ*sizeof(DATA_TYPE)){
    C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
    C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
  }else{
    cudaMallocHost((void**)&C, NI*NJ*sizeof(DATA_TYPE));
    cudaMallocHost((void**)&C_outputFromGpu, NI*NJ*sizeof(DATA_TYPE));
  }

  printf("<< Matrix-multiply C=alpha.A.B+beta.C >>\n");
  printf("NI: %d\n", NI);
  printf("NJ: %d\n", NJ);
  printf("NK: %d\n", NK);

  init(A, B, C, C_outputFromGpu);

  t_start = rtclock();

  GPU__gemm(A, B, C_outputFromGpu);

  update(C_outputFromGpu);

  t_end = rtclock();
  printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

  // t_start = rtclock();
  // gemm(A, B, C);
  // t_end = rtclock();
  // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  // compareResults(C, C_outputFromGpu);

  if(threshold > NI*NK*sizeof(DATA_TYPE)){
    free(A);
  }else{
    cudaFreeHost(A);   
  }

  if(threshold > NK*NJ*sizeof(DATA_TYPE)){
    free(B);
  }else{
    cudaFreeHost(B);
  }

  if(threshold > NI*NJ*sizeof(DATA_TYPE)){
    free(C);
    free(C_outputFromGpu);
  }else{
    cudaFreeHost(C);
    cudaFreeHost(C_outputFromGpu);
  }

  return 0;
}

