#include <malloc.h>
#include "RunTime.h"
STATE_CONSTR StConstrTarget;
/**
 * 3mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU 
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05


#ifdef _DEBUG_1
/* Problem size. */
# define NI 6000
# define NJ 6000
# define NK 6000
# define NL 6000
# define NM 6000

#elif _DEBUG_2
/* Problem size. */
# define NI 3000
# define NJ 3000
# define NK 3000
# define NL 3000
# define NM 3000

#else
/* Problem size. */
# define NI 1500
# define NJ 1500
# define NK 1500
# define NL 1500
# define NM 1500
#endif

# define GPU_DEVICE 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
  int i, j;

  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NK; j++)
	{
	  A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
	}
    }
  
  for (i = 0; i < NK; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
	}
    }
  
  for (i = 0; i < NJ; i++)
    {
      for (j = 0; j < NM; j++)
	{
	  C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
	}
    }
  
  for (i = 0; i < NM; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
	}
    }
}

void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
  int i,j,fail;
  fail = 0;

  for (i=0; i < NI; i++)
    {
      for (j=0; j < NL; j++)
	{
	  if (percentDiff(G[i*NL + j], G_outputFromGpu[i*NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
	    {
	      fail++;				
	    }
	}
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
  int i,j,k;
	
  /* E := A*B */
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  E[i*NJ + j] = 0;
	  for (k = 0; k < NK; ++k)
	    {
	      E[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
	    }
	}
    }
		
  /* F := C*D */
  for (i = 0; i < NJ; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  F[i*NL + j] = 0;
	  for (k = 0; k < NM; ++k)
	    {
	      F[i*NL + j] += C[i*NM + k] * D[k*NL + j];
	    }
	}
    }

  /* G := E*F */
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  G[i*NL + j] = 0;
	  for (k = 0; k < NJ; ++k)
	    {
	      G[i*NL + j] += E[i*NJ + k] * F[k*NL + j];
	    }
	}
    }
}

void GPU__mm3(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
  int i,j,k;
	
  /* E := A*B */
  #pragma omp target teams distribute parallel for map(tofrom: i, j, k) collapse(2)
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  E[i*NJ + j] = 0;
	  for (k = 0; k < NK; ++k)
	    {
	      E[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
	    }
	}
    }
    OAOStTrans( E, StConstrTarget.init(5, 4) );
    
  
  /* F := C*D */
  #pragma omp target teams distribute parallel for map(tofrom: i, j, k) collapse(2)
  for (i = 0; i < NJ; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  F[i*NL + j] = 0;
	  for (k = 0; k < NM; ++k)
	    {
	      F[i*NL + j] += C[i*NM + k] * D[k*NL + j];
	    }
	}
    }
    OAOStTrans( F, StConstrTarget.init(5, 4) );
    

  /* G := E*F */
  OAODataTrans( E, StConstrTarget.init(7, 5) );
  OAODataTrans( F, StConstrTarget.init(7, 5) );
  
  #pragma omp target teams distribute parallel for map(tofrom: i, j, k) collapse(2)
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  G[i*NL + j] = 0;
	  for (k = 0; k < NJ; ++k)
	    {
	      G[i*NL + j] += E[i*NJ + k] * F[k*NL + j];
	    }
	}
    }
    OAOStTrans( G, StConstrTarget.init(5, 4) );
    
  return;
}

inline void update(DATA_TYPE* G_outputFromGpu)
{
  DATA_TYPE cc = G_outputFromGpu[0];
}

//int main(int argc, char** argv)
int main()
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* C;
  DATA_TYPE* D;
  DATA_TYPE* E;
  DATA_TYPE* F;
  DATA_TYPE* G;
  DATA_TYPE* G_outputFromGpu;

  A = (DATA_TYPE*)OAOMalloc(NI*NK*sizeof(DATA_TYPE));
  B = (DATA_TYPE*)OAOMalloc(NK*NJ*sizeof(DATA_TYPE));
  C = (DATA_TYPE*)OAOMalloc(NJ*NM*sizeof(DATA_TYPE));
  D = (DATA_TYPE*)OAOMalloc(NM*NL*sizeof(DATA_TYPE));
  E = (DATA_TYPE*)OAOMalloc(NI*NJ*sizeof(DATA_TYPE));
  F = (DATA_TYPE*)OAOMalloc(NJ*NL*sizeof(DATA_TYPE));
  G = (DATA_TYPE*)OAOMalloc(NI*NL*sizeof(DATA_TYPE));
  G_outputFromGpu = (DATA_TYPE*)OAOMalloc(NI*NL*sizeof(DATA_TYPE));

  printf("<< Linear Algebra: 3 Matrix Multiplications (E=A.B; F=C.D; G=E.F) >>\n");

  
  OAODataTrans( A, StConstrTarget.init(7, 3) );
  OAODataTrans( B, StConstrTarget.init(7, 3) );
  OAODataTrans( C, StConstrTarget.init(7, 3) );
  OAODataTrans( D, StConstrTarget.init(7, 3) );
  init_array(A, B, C, D);

  t_start = rtclock();
  
  OAODataTrans( A, StConstrTarget.init(7, 5) );
  OAODataTrans( B, StConstrTarget.init(7, 5) );
  OAODataTrans( C, StConstrTarget.init(7, 5) );
  OAODataTrans( D, StConstrTarget.init(7, 5) );
  OAODataTrans( E, StConstrTarget.init(7, 5) );
  OAODataTrans( F, StConstrTarget.init(7, 5) );
  OAODataTrans( G_outputFromGpu, StConstrTarget.init(7, 5) );
  GPU__mm3(A, B, C, D, E, F, G_outputFromGpu);
  
  
  OAODataTrans( G_outputFromGpu, StConstrTarget.init(7, 3) );
  update(G_outputFromGpu);
  
  t_end = rtclock();	

  printf("GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  
  OAODataTrans( A, StConstrTarget.init(7, 3) );
  OAODataTrans( B, StConstrTarget.init(7, 3) );
  OAODataTrans( C, StConstrTarget.init(7, 3) );
  OAODataTrans( D, StConstrTarget.init(7, 3) );
  OAODataTrans( E, StConstrTarget.init(7, 3) );
  OAODataTrans( F, StConstrTarget.init(7, 3) );
  //mm3_cpu(A, B, C, D, E, F, G);
  t_end = rtclock();

  printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  
  OAODataTrans( G_outputFromGpu, StConstrTarget.init(7, 3) );
  compareResults(G, G_outputFromGpu);

  
  OAODataTrans( A, StConstrTarget.init(2, 2) );
  OAOFree(A);
  
  OAODataTrans( B, StConstrTarget.init(2, 2) );
  OAOFree(B);
  
  OAODataTrans( C, StConstrTarget.init(2, 2) );
  OAOFree(C);
  
  OAODataTrans( D, StConstrTarget.init(2, 2) );
  OAOFree(D);
  
  OAODataTrans( E, StConstrTarget.init(2, 2) );
  OAOFree(E);
  
  OAODataTrans( F, StConstrTarget.init(2, 2) );
  OAOFree(F);
  OAOFree(G);
  
  OAODataTrans( G_outputFromGpu, StConstrTarget.init(2, 2) );
  OAOFree(G_outputFromGpu);

  return 0;

OAODataTrans( A, StConstrTarget.init(2, 2) );
OAODataTrans( B, StConstrTarget.init(2, 2) );
OAODataTrans( C, StConstrTarget.init(2, 2) );
OAODataTrans( D, StConstrTarget.init(2, 2) );
OAODataTrans( E, StConstrTarget.init(2, 2) );
OAODataTrans( F, StConstrTarget.init(2, 2) );
OAODataTrans( G_outputFromGpu, StConstrTarget.init(2, 2) );
}

