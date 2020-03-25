/**
 * 2mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
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

//define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1


#ifdef _DEBUG_1
/* Problem size. */
# define NI 3000
# define NJ 3000
# define NK 3000
# define NL 3000

#elif _DEBUG_2
/* Problem size. */
# define NI 1500
# define NJ 1500
# define NK 1500
# define NL 1500

#else
/* Problem size. */
# define NI 750
# define NJ 750
# define NK 750
# define NL 750
#endif


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
  int i, j;

  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NK; j++)
	{
	  A[i*NI + j] = ((DATA_TYPE) i*j) / NI;
	}
    }
  
  for (i = 0; i < NK; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  B[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
	}
    }
  
  for (i = 0; i < NL; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  C[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
	}
    }
  
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;	
	}
    }
}

void compareResults(DATA_TYPE *E, DATA_TYPE *E_GPU)
{
  int i,j,fail;
  fail = 0;

  for (i=0; i < NL; i++)
    {
      for (j=0; j < NI; j++)
	{
	  if (percentDiff(E[i*NI + j], E_GPU[i*NI + j]) > ERROR_THRESHOLD)
	    {
	      fail++;
	    }
	}
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
  int i, j, k;

  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  C[i*NJ + j] = 0.0;
	  for (k = 0; k < NK; ++k)
	    {
	      C[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
	    }
	}
    }
  
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  E[i*NL + j] = 0.0;
	  for (k = 0; k < NJ; ++k)
	    {
	      E[i*NL + j] += C[i*NJ + k] * D[k*NL + j];
	    }
	}
    }
}

void GPU__mm2(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
  int i, j, k;

  #pragma omp parallel for collapse(2)
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  C[i*NJ + j] = 0.0;
	  for (k = 0; k < NK; ++k)
	    {
	      C[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
	    }
	}
    }
	
  #pragma omp parallel for collapse(2)
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  E[i*NL + j] = 0.0;
	  for (k = 0; k < NJ; ++k)
	    {
	      E[i*NL + j] += C[i*NJ + k] * D[k*NL + j];
	    }
	}
    }
  return;
}


inline void update(DATA_TYPE* E_GPU)
{
  DATA_TYPE ee = E_GPU[0];
}

//int main(int argc, char** argv)
int main()
{
  double t_start, t_end, t_start_GPU, t_end_GPU;

  DATA_TYPE* C;
  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* D;
  DATA_TYPE* E;
  DATA_TYPE* E_GPU;

  C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
  A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
  D = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
  E = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
  E_GPU = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

  printf("<< Linear Algebra: 2 Matrix Multiplications (D=A.B; E=C.D) >>\n");

  init_array(A, B, C, D);

  // t_start_GPU = rtclock();
  // GPU__mm2(A, B, C, D, E_GPU);
  
  // update(E_GPU);
  
  // t_end_GPU = rtclock();
  // printf("GPU Runtime: %0.6lfs\n", t_end_GPU - t_start_GPU);	

  t_start = rtclock();
  mm2_cpu(A, B, C, D, E);
  t_end = rtclock();
  printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  //compareResults(E, E_GPU);

  free(C);
  free(A);
  free(B);
  free(D);
  free(E);
  free(E_GPU);

  return 0;
}

