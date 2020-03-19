#include <malloc.h>
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

void init_array(DATA_TYPE *A, int A_LEN_, DATA_TYPE *p, int p_LEN_, DATA_TYPE *r, int r_LEN_)
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

void compareResults(DATA_TYPE* s, int s_LEN_, DATA_TYPE* s_outputFromGpu, int s_outputFromGpu_LEN_, DATA_TYPE* q, int q_LEN_, DATA_TYPE* q_outputFromGpu, int q_outputFromGpu_LEN_)
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


void bicg_cpu(DATA_TYPE* A, int A_LEN_, DATA_TYPE* r, int r_LEN_, DATA_TYPE* s, int s_LEN_, DATA_TYPE* p, int p_LEN_, DATA_TYPE* q, int q_LEN_)
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

void GPU__bicg(DATA_TYPE* A, int A_LEN_, DATA_TYPE* r, int r_LEN_, DATA_TYPE* s, int s_LEN_, DATA_TYPE* p, int p_LEN_, DATA_TYPE* q, int q_LEN_)
{
  int i, j;
	
  for (i = 0; i < NY; i++)
    {
      s[i] = 0.0;
    }

  {
  #pragma omp target update to( s[:s_LEN_] )
  
	  #pragma omp target teams distribute parallel for 
	  for (j = 0; j < NY; j++)
	  {
		for (i = 0; i < NX; i++)
	  	{
	    		s[j] = s[j] + r[i] * A[i*NY + j];
	  	}
	  }

	   #pragma omp target teams distribute parallel for 
	   for (i = 0; i < NX; i++)
	   {
		q[i] = 0.0;
		for (j = 0; j < NY; j++)
	  	{
	    		q[i] = q[i] + A[i*NY + j] * p[j];
	  	}
	   } 
   }
  return;
}

inline void update(DATA_TYPE* s_GPU, int s_GPU_LEN_, DATA_TYPE* q_GPU, int q_GPU_LEN_)
{
  DATA_TYPE cc = s_GPU[0];
  DATA_TYPE qq = q_GPU[0];
}

//int main(int argc, char** argv)
int main()
{
  double t_start, t_end;

  DATA_TYPE* A; int A_LEN_;
  DATA_TYPE* r; int r_LEN_;
  DATA_TYPE* s; int s_LEN_;
  DATA_TYPE* p; int p_LEN_;
  DATA_TYPE* q; int q_LEN_;
  DATA_TYPE* s_GPU; int s_GPU_LEN_;
  DATA_TYPE* q_GPU; int q_GPU_LEN_;
 	
  A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE)); A_LEN_ = malloc_usable_size( A ) / sizeof( DATA_TYPE );
  r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE)); r_LEN_ = malloc_usable_size( r ) / sizeof( DATA_TYPE );
  s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE)); s_LEN_ = malloc_usable_size( s ) / sizeof( DATA_TYPE );
  p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE)); p_LEN_ = malloc_usable_size( p ) / sizeof( DATA_TYPE );
  q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE)); q_LEN_ = malloc_usable_size( q ) / sizeof( DATA_TYPE );
  s_GPU = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE)); s_GPU_LEN_ = malloc_usable_size( s_GPU ) / sizeof( DATA_TYPE );
  q_GPU = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE)); q_GPU_LEN_ = malloc_usable_size( q_GPU ) / sizeof( DATA_TYPE );

  printf("<< BiCG Sub Kernel of BiCGStab Linear Solver >>\n");

  
  #pragma omp target enter data map( alloc: A[:A_LEN_], r[:r_LEN_], p[:p_LEN_] )
  init_array(A, A_LEN_, p, p_LEN_, r, r_LEN_);

  t_start = rtclock();
  
  #pragma omp target enter data map( to: s_GPU[:s_GPU_LEN_], q_GPU[:q_GPU_LEN_] )
  
  #pragma omp target update to( A[:A_LEN_], r[:r_LEN_], p[:p_LEN_] )
  GPU__bicg(A, A_LEN_, r, r_LEN_, s_GPU, s_GPU_LEN_, p, p_LEN_, q_GPU, q_GPU_LEN_);
  
  
  #pragma omp target update from( s_GPU[:s_GPU_LEN_], q_GPU[:q_GPU_LEN_] )
  update(s_GPU, s_GPU_LEN_, q_GPU, q_GPU_LEN_);
  
  t_end = rtclock();

  printf("GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  bicg_cpu(A, A_LEN_, r, r_LEN_, s, s_LEN_, p, p_LEN_, q, q_LEN_);
  t_end = rtclock();

  printf( "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(s, s_LEN_, s_GPU, s_GPU_LEN_, q, q_LEN_, q_GPU, q_GPU_LEN_);

  
  #pragma omp target exit data map( delete: A[:A_LEN_] )
  free(A);
  
  #pragma omp target exit data map( delete: r[:r_LEN_] )
  free(r);
  free(s);
  
  #pragma omp target exit data map( delete: p[:p_LEN_] )
  free(p);
  free(q);
  
  #pragma omp target exit data map( delete: s_GPU[:s_GPU_LEN_] )
  free(s_GPU);
  
  #pragma omp target exit data map( delete: q_GPU[:q_GPU_LEN_] )
  free(q_GPU);

  return 0;
}

