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


#ifdef _DEBUG_1
/* Problem size. */
#define NX 32768
#define NY 32768

#elif _DEBUG_2
/* Problem size. */
#define NX 16384
#define NY 16384

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

void GPU__bicg(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
  int i, j;
	
  for (i = 0; i < NY; i++)
    {
      s[i] = 0.0;
    }

  {
	  #pragma omp parallel for
	  for (j = 0; j < NY; j++)
	  {
		for (i = 0; i < NX; i++)
	  	{
	    		s[j] = s[j] + r[i] * A[i*NY + j];
	  	}
	  }

	   #pragma omp parallel for
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

inline void update(DATA_TYPE* s_GPU, DATA_TYPE* q_GPU)
{
  DATA_TYPE cc = s_GPU[0];
  DATA_TYPE qq = q_GPU[0];
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
 	
  A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
  r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
  s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
  p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
  q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
  s_GPU = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
  q_GPU = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

  printf("<< BiCG Sub Kernel of BiCGStab Linear Solver >>\n");
  printf("NX: %d\n", NX);
  printf("NY: %d\n", NY);

  init_array(A, p, r);

  t_start = rtclock();
  GPU__bicg(A, r, s_GPU, p, q_GPU);
  
  update(s_GPU, q_GPU);
  
  t_end = rtclock();

    printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

  // t_start = rtclock();
  // bicg_cpu(A, r, s, p, q);
  // t_end = rtclock();

  // printf( "CPU Runtime: %0.6lfs\n", t_end - t_start);

  // compareResults(s, s_GPU, q, q_GPU);

  free(A);
  free(r);
  free(s);
  free(p);
  free(q);
  free(s_GPU);
  free(q_GPU);

  return 0;
}

