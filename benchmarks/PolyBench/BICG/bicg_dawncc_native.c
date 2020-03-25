/**
 * bicg.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
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
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

float absVal(float a)
{
	if(a < 0)
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

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.7

/* Problem size. */
#define NX 8192
#define NY 8192

#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r)
{
  int i, j;

  for (i = 0; i < NX; i++)
    {
      r[i] = i * M_PI;
      for (j = 0; j < NY; j++)
	{
	  A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
	}
    }
  
  for (i = 0; i < NY; i++)
    {
      p[i] = i * M_PI;
    }
}

void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
  int i,fail;
  fail = 0;

  // Compare s with s_cuda
  for (i=0; i<NX; i++)
    {
      if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
	{
	  fail++;
	}
    }
  
  for (i=0; i<NY; i++)
    {
      if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
	{
	  fail++;
	}		
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
  int i,j;
	
  for (i = 0; i < NY; i++)
    {
      s[i] = 0.0;
    }
  
  for (i = 0; i < NX; i++)
    {
      q[i] = 0.0;
      for (j = 0; j < NY; j++)
	{
	  s[j] = s[j] + r[i] * A[i*NY + j];
	  q[i] = q[i] + A[i*NY + j] * p[j];
	}
    }
}

void GPU__bicg(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p,
               DATA_TYPE *q) {
  int i, j;

  for (i = 0; i < NY; i++)
    {
      s[i] = 0.0;
    }

  {
    char RST_AI2 = 0;
    RST_AI2 |= !(((void*) (A + 0) > (void*) (r + 8191))
    || ((void*) (r + 0) > (void*) (A + 67108863)));
    RST_AI2 |= !(((void*) (A + 0) > (void*) (s + 8191))
    || ((void*) (s + 0) > (void*) (A + 67108863)));
    RST_AI2 |= !(((void*) (r + 0) > (void*) (s + 8191))
    || ((void*) (s + 0) > (void*) (r + 8191)));
    #pragma omp target data map(to: A[0:67108864],r[0:8192]) map(tofrom: s[0:8192]) if(!RST_AI2)
    {
    #pragma omp target parallel for if(!RST_AI2)
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        s[j] = s[j] + r[i] * A[i * NY + j];
      }
    }
    }

    char RST_AI3 = 0;
    RST_AI3 |= !(((void*) (A + 0) > (void*) (p + 8191))
    || ((void*) (p + 0) > (void*) (A + 67108863)));
    RST_AI3 |= !(((void*) (A + 0) > (void*) (q + 8191))
    || ((void*) (q + 0) > (void*) (A + 67108863)));
    RST_AI3 |= !(((void*) (p + 0) > (void*) (q + 8191))
    || ((void*) (q + 0) > (void*) (p + 8191)));
    #pragma omp target data map(to: A[0:67108864],p[0:8192]) map(tofrom: q[0:8192]) if(!RST_AI3)
    {
    #pragma omp target parallel for if(!RST_AI3)
    for (i = 0; i < NX; i++) {
      q[i] = 0.0;
      for (j = 0; j < NY; j++) {
        q[i] = q[i] + A[i * NY + j] * p[j];
      }
    }
  }
  }
  return;
}


//int main(int argc, char** argv)
int main()
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* r;
  DATA_TYPE* s;
  DATA_TYPE* p;
  DATA_TYPE* q;
  DATA_TYPE* s_GPU;
  DATA_TYPE* q_GPU;
 	
  const unsigned long long int threshold = (unsigned long long int)0xFFFFFFFFFFFFFFFF; // 128 * 1024;
  if(threshold > NX*NY*sizeof(DATA_TYPE)){
      A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
  }else{
      cudaMallocHost((void**)&A, NX*NY*sizeof(DATA_TYPE));
  }
  if(threshold > NX*sizeof(DATA_TYPE)){
      r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
      s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
      p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
      q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
      s_GPU = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
      q_GPU = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
  }else{
      cudaMallocHost((void**)&r, NX*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&s, NY*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&p, NY*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&q, NX*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&s_GPU, NY*sizeof(DATA_TYPE));
      cudaMallocHost((void**)&q_GPU, NX*sizeof(DATA_TYPE));
  }


  printf("<<DawnCC BiCG Sub Kernel of BiCGStab Linear Solver >>\n");
  printf("NX: %d\n", NX);
  printf("NY: %d\n", NY);

  init_array(A, p, r);

  t_start = rtclock();
  GPU__bicg(A, r, s_GPU, p, q_GPU);
  t_end = rtclock();

    printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

  // t_start = rtclock();
  // bicg_cpu(A, r, s, p, q);
  // t_end = rtclock();

  // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  // compareResults(s, s_GPU, q, q_GPU);

  if(threshold > NX*NY*sizeof(DATA_TYPE)){
      free(A);
  }else{
      cudaFreeHost(A);
  }
  if(threshold > NX*sizeof(DATA_TYPE)){
      free(r);
      free(s);
      free(p);
      free(q);
      free(s_GPU);
      free(q_GPU);
  }else{
      cudaFreeHost(r);
      cudaFreeHost(s);
      cudaFreeHost(p);
      cudaFreeHost(q);
      cudaFreeHost(s_GPU);
      cudaFreeHost(q_GPU);
  }

  return 0;
}

