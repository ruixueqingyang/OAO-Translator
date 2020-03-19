/**
 * correlation.c This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *  	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>

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
#define ERROR_THRESHOLD 1.05

#define GPU_DEVICE 1

#ifdef _DEBUG_1
/* Problem size */
#define N 4096
#define M 4096

#elif _DEBUG_2
/* Problem size */
#define N 2048
#define M 2048

#else
/* Problem size */
#define N 1024
#define M 1024
#endif

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *data)
{
  int i, j;

  for (i = 0; i < (M + 1); i++)
  {
    for (j = 0; j < (N + 1); j++)
    {
      data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / (M + 1);
    }
  }
}

void correlation(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev, DATA_TYPE *symmat)
{
  int i, j, j1, j2;

  // Determine mean of column vectors of input data matrix
  for (j = 1; j < (M + 1); j++)
  {
    mean[j] = 0.0;

    for (i = 1; i < (N + 1); i++)
    {
      mean[j] += data[i * (M + 1) + j];
    }

    mean[j] /= (DATA_TYPE)FLOAT_N;
  }

  // Determine standard deviations of column vectors of data matrix.
  for (j = 1; j < (M + 1); j++)
  {
    stddev[j] = 0.0;

    for (i = 1; i < (N + 1); i++)
    {
      stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
    }

    stddev[j] /= FLOAT_N;
    stddev[j] = sqrt_of_array_cell(stddev, j);
    stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
  }

  //i - threadIdx.x, j = threadIdx.y
  // Center and reduce the column vectors.
  for (i = 1; i < (N + 1); i++)
  {
    for (j = 1; j < (M + 1); j++)
    {
      data[i * (M + 1) + j] -= mean[j];
      data[i * (M + 1) + j] /= (sqrt(FLOAT_N) * stddev[j]);
    }
  }

  // Calculate the m * m correlation matrix.
  for (j1 = 1; j1 < M; j1++)
  {
    symmat[j1 * (M + 1) + j1] = 1.0;

    for (j2 = j1 + 1; j2 < (M + 1); j2++)
    {
      symmat[j1 * (M + 1) + j2] = 0.0;

      for (i = 1; i < (N + 1); i++)
      {
        symmat[j1 * (M + 1) + j2] += (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
      }

      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }

  symmat[M * (M + 1) + M] = 1.0;
}

void GPU__correlation(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev, DATA_TYPE *symmat)
{
  int i, j, k;

// Determine mean of column vectors of input data matrix
#pragma omp parallel for
  for (j = 1; j < (M + 1); j++)
  {
    mean[j] = 0.0;
    for (i = 1; i < (N + 1); i++)
    {
      mean[j] += data[i * (M + 1) + j];
    }
    mean[j] /= (DATA_TYPE)FLOAT_N;
  }

  // Determine standard deviations of column vectors of data matrix.
  for (j = 1; j < (M + 1); j++)
  {
    stddev[j] = 0.0;
    for (i = 1; i < (N + 1); i++)
    {
      stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
    }

    stddev[j] /= FLOAT_N;
    stddev[j] = sqrt(stddev[j]);
    if (stddev[j] <= EPS)
    {
      stddev[j] = 1.0;
    }
  }

  // Center and reduce the column vectors.
  for (i = 1; i < (N + 1); i++)
  {
    for (j = 1; j < (M + 1); j++)
    {
      data[i * (M + 1) + j] -= mean[j];
      data[i * (M + 1) + j] /= (sqrt(FLOAT_N) * stddev[j]);
    }
  }

// Calculate the m * m correlation matrix.
#pragma omp parallel for collapse(1)
  for (k = 1; k < M; k++)
  {
    symmat[k * (M + 1) + k] = 1.0;
    for (j = k + 1; j < (M + 1); j++)
    {
      symmat[k * (M + 1) + j] = 0.0;
      for (int i = 1; i < (N + 1); i++)
      {
        symmat[k * (M + 1) + j] += (data[i * (M + 1) + k] * data[i * (M + 1) + j]);
      }
      symmat[j * (M + 1) + k] = symmat[k * (M + 1) + j];
    }
  }

  symmat[M * (M + 1) + M] = 1.0;
  return;
}

void compareResults(DATA_TYPE *symmat, DATA_TYPE *symmat_outputFromGpu)
{
  int i, j, fail;
  fail = 0;

  for (i = 1; i < (M + 1); i++)
  {
    for (j = 1; j < (N + 1); j++)
    {
      if (percentDiff(symmat[i * (N + 1) + j], symmat_outputFromGpu[i * (N + 1) + j]) > ERROR_THRESHOLD)
      {
        fail++;
        //printf("i: %d j: %d\n1: %f 2: %f\n", i, j, symmat[i*N + j], symmat_GPU[i*N + j]);
      }
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

int main()
{
  double t_start, t_end;

  DATA_TYPE *data;
  DATA_TYPE *mean;
  DATA_TYPE *stddev;
  DATA_TYPE *symmat;
  DATA_TYPE *symmat_GPU;

  data = (DATA_TYPE *)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  mean = (DATA_TYPE *)malloc((M + 1) * sizeof(DATA_TYPE));
  stddev = (DATA_TYPE *)malloc((M + 1) * sizeof(DATA_TYPE));
  symmat = (DATA_TYPE *)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  symmat_GPU = (DATA_TYPE *)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));

  printf("<< Correlation Computation >>\n");
  printf("N: %d\n", N);
  printf("M: %d\n", M);

  init_arrays(data);

  t_start = rtclock();
  GPU__correlation(data, mean, stddev, symmat_GPU);
  t_end = rtclock();

    printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

  // init_arrays(data);

  // t_start = rtclock();
  // correlation(data, mean, stddev, symmat);
  // t_end = rtclock();

  // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  // compareResults(symmat, symmat_GPU);

  free(data);
  free(mean);
  free(stddev);
  free(symmat);
  free(symmat_GPU);

  return 0;
}
