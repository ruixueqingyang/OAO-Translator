#include <malloc.h>
/**
 * atax.c: This file was adapted from PolyBench/GPU 1.0 test suite
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
#include <omp.h>
#include <cuda_runtime.h>

//polybenchUtilFuncts.h
//Scott Grauer-Gray (sgrauerg@gmail.com)
//Functions used across hmpp codes

#ifndef POLYBENCH_UTIL_FUNCTS_H
#define POLYBENCH_UTIL_FUNCTS_H

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a)
{
  if (a < 0)
  {
    return (a * -1);
  }
  else
  {
    return a;
  }
}

float percentDiff(double val1, double val2)
{
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
  {
    return 0.0f;
  }

  else
  {
    return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
  }
}

#endif //POLYBENCH_UTIL_FUNCTS_H

int main();
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5


#ifdef _DEBUG_1
/* Problem size. */
#define NX 32768
#define NY 32768

#elif _DEBUG_2
/* Problem size. */
#define NX 20480
#define NY 20480

#else
/* Problem size. */
#define NX 8192
#define NY 8192
#endif


#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
  int i, j;

  for (i = 0; i < NX; i++)
  {
    x[i] = i * M_PI;
    for (j = 0; j < NY; j++)
    {
      A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
}

void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
  int i, fail;
  fail = 0;

  for (i = 0; i < NY; i++)
  {
    if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
    {
      fail++;
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void atax_cpu(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
  int i, j;

  for (i = 0; i < NY; i++)
  {
    y[i] = 0;
  }

  for (i = 0; i < NX; i++)
  {
    tmp[i] = 0;

    for (j = 0; j < NY; j++)
    {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }

    for (j = 0; j < NY; j++)
    {
      y[j] = y[j] + A[i * NY + j] * tmp[i];
    }
  }
}

void GPU__atax(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
  int i, j;

  for (i = 0; i < NY; i++)
  {
    y[i] = 0;
  }
  int Length1 = NX * NY;
  int Length2 = NX;
#pragma omp target data map( tofrom: A[:Length1], x[:Length2], tmp[:Length2] )
{
#pragma omp target teams distribute parallel for map(tofrom: i, j)
  for (i = 0; i < NX; i++)
  {
    tmp[i] = 0;
    for (j = 0; j < NY; j++)
    {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }
  }
}
  

//Note that the Loop has been reversed
#pragma omp target data map( tofrom: A[:Length1], y[:Length2], tmp[:Length2] )
{
#pragma omp target teams distribute parallel for map(tofrom: j, i)
  for (j = 0; j < NY; j++)
    for (i = 0; i < NX; i++)
    {
      {
        y[j] = y[j] + A[i * NY + j] * tmp[i];
      }
    }
}
  return;
}

inline void update(DATA_TYPE *y_outputFromGpu)
{
  DATA_TYPE yy = y_outputFromGpu[0];
}

//int main(int argc, char** argv)
int main()
{
  double t_start, t_end;
  DATA_TYPE *A;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  const unsigned long long int threshold = (unsigned long long int)128 * 1024;
  if(threshold > NX*NY*sizeof(DATA_TYPE)){
      A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
  }else{
      cudaMallocHost((void**)&A, NX*NY*sizeof(DATA_TYPE));
  }
  if(threshold > NY*sizeof(DATA_TYPE)){
      x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
      y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
      y_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
      tmp = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
  }else{
      cudaMallocHost((void**)&x, NY*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&y, NY*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&y_outputFromGpu, NY*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&tmp, NY*sizeof(DATA_TYPE));
  }


  printf("<< Matrix Transpose and Vector Multiplication >>\n");
  printf("NX: %d\n", NX);
  printf("NY: %d\n", NY);

  init_array(x, A);

  t_start = rtclock();
  GPU__atax(A, x, y_outputFromGpu, tmp);

  update(y_outputFromGpu);

  t_end = rtclock();
  printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

  // t_start = rtclock();
  // atax_cpu(A, x, y, tmp);
  // t_end = rtclock();
  // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  // compareResults(y, y_outputFromGpu);

  if(threshold > NX*NY*sizeof(DATA_TYPE)){
    free(A);
  }else{
    cudaFreeHost(A);
  }

  if(threshold > NY*sizeof(DATA_TYPE)){
    free(x);
    free(y);
    free(y_outputFromGpu);
    free(tmp);
  }else{
    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFreeHost(y_outputFromGpu);
    cudaFreeHost(tmp);
  }

  return 0;

}
