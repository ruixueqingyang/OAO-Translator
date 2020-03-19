/**
 * gesummv.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1

/* Problem size */
#define N 32768

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
             DATA_TYPE *tmp) {
  int i, j;

  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}

void GPU__gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
                  DATA_TYPE *tmp) {
  int i, j;
  double s1, s2, e1, e2; 

  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (A + 0) > (void*) (B + 1073741824))
  || ((void*) (B + 0) > (void*) (A + 1073741824)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (tmp + 32768))
  || ((void*) (tmp + 0) > (void*) (A + 1073741824)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (x + 32768))
  || ((void*) (x + 0) > (void*) (A + 1073741824)));
  RST_AI1 |= !(((void*) (A + 0) > (void*) (y + 32768))
  || ((void*) (y + 0) > (void*) (A + 1073741824)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (tmp + 32768))
  || ((void*) (tmp + 0) > (void*) (B + 1073741824)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (x + 32768))
  || ((void*) (x + 0) > (void*) (B + 1073741824)));
  RST_AI1 |= !(((void*) (B + 0) > (void*) (y + 32768))
  || ((void*) (y + 0) > (void*) (B + 1073741824)));
  RST_AI1 |= !(((void*) (tmp + 0) > (void*) (x + 32768))
  || ((void*) (x + 0) > (void*) (tmp + 32768)));
  RST_AI1 |= !(((void*) (tmp + 0) > (void*) (y + 32768))
  || ((void*) (y + 0) > (void*) (tmp + 32768)));
  RST_AI1 |= !(((void*) (x + 0) > (void*) (y + 32768))
  || ((void*) (y + 0) > (void*) (x + 32768)));
  s1 = rtclock();
  #pragma omp target enter data map(to: A[0:1073741824],B[0:1073741824],x[0:32768],tmp[0:32768],y[0:32768]) if(!RST_AI1)
  e1 = rtclock();
  {
  #pragma omp target teams distribute parallel for if(!RST_AI1)
  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
  }
  s2 = rtclock();
  #pragma omp target update from(tmp[0:32768],y[0:32768])
  e2 = rtclock();
  //printf("Extra cost time(s) is: %0.6lf\n", e1 - s1);

  printf("Host to Device number of times is: %d\n", 5);
  printf("Host to Device number of byte is: %ld\n", (1073741824+32768+32768+32768) * sizeof(DATA_TYPE) + +1073741824 * sizeof(DATA_TYPE));
  printf("Host to Device spend of time(s) is: %0.6lf\n", e1 - s1);
  
  printf("Device to Host number of times is: %d\n", 2);
  printf("Device to Host number of byte is: %ld\n", (32768+32768) * sizeof(DATA_TYPE));
  printf("Device to Host spend of time(s) is: %0.6lf\n", e2 - s2);
 
  return;
}

void init(DATA_TYPE *A, DATA_TYPE *x) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void compareResults(DATA_TYPE *y, DATA_TYPE *y_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < (N); i++) {
    if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

inline void update(DATA_TYPE *y_outputFromGpu) {
  DATA_TYPE yy = y_outputFromGpu[0];
}

// int main(int argc, char *argv[])
int main() {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  printf("<< Scalar, Vector and Matrix Multiplication >>\n");
  printf("Data size N: %d\n", N);

  init(A, x);

  t_start = rtclock();
  GPU__gesummv(A, B, x, y_outputFromGpu, tmp);

  update(y_outputFromGpu);

  t_end = rtclock();
  printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

  // t_start = rtclock();
  // gesummv(A, B, x, y, tmp);
  // t_end = rtclock();
  // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  // compareResults(y, y_outputFromGpu);

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}

