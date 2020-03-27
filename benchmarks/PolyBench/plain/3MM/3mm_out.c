#include <malloc.h>
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

/* Problem size. */
# define NI 1500
# define NJ 1500
# define NK 1500
# define NL 1500
# define NM 1500

# define GPU_DEVICE 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, int A_LEN_, DATA_TYPE* B, int B_LEN_, DATA_TYPE* C, int C_LEN_, DATA_TYPE* D, int D_LEN_)
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

void compareResults(DATA_TYPE *G, int G_LEN_, DATA_TYPE *G_outputFromGpu, int G_outputFromGpu_LEN_)
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

void mm3_cpu(DATA_TYPE *A, int A_LEN_, DATA_TYPE *B, int B_LEN_, DATA_TYPE *C, int C_LEN_, DATA_TYPE *D, int D_LEN_, DATA_TYPE *E, int E_LEN_, DATA_TYPE *F, int F_LEN_, DATA_TYPE *G, int G_LEN_)
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

void GPU__mm3(DATA_TYPE *A, int A_LEN_, DATA_TYPE *B, int B_LEN_, DATA_TYPE *C, int C_LEN_, DATA_TYPE *D, int D_LEN_, DATA_TYPE *E, int E_LEN_, DATA_TYPE *F, int F_LEN_, DATA_TYPE *G, int G_LEN_)
{
  int i,j,k;
	
  /* E := A*B */
  #pragma omp target teams distribute parallel for  collapse(2)
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
  #pragma omp target teams distribute parallel for  collapse(2)
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
  #pragma omp target teams distribute parallel for  collapse(2)
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
  return;
}

inline void update(DATA_TYPE* G_outputFromGpu, int G_outputFromGpu_LEN_)
{
  DATA_TYPE cc = G_outputFromGpu[0];
}

//int main(int argc, char** argv)
int main()
{
  double t_start, t_end;

  DATA_TYPE* A; int A_LEN_;
  DATA_TYPE* B; int B_LEN_;
  DATA_TYPE* C; int C_LEN_;
  DATA_TYPE* D; int D_LEN_;
  DATA_TYPE* E; int E_LEN_;
  DATA_TYPE* F; int F_LEN_;
  DATA_TYPE* G; int G_LEN_;
  DATA_TYPE* G_outputFromGpu; int G_outputFromGpu_LEN_;

  A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); A_LEN_ = malloc_usable_size( A ) / sizeof( DATA_TYPE );
  B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE)); B_LEN_ = malloc_usable_size( B ) / sizeof( DATA_TYPE );
  C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE)); C_LEN_ = malloc_usable_size( C ) / sizeof( DATA_TYPE );
  D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE)); D_LEN_ = malloc_usable_size( D ) / sizeof( DATA_TYPE );
  E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); E_LEN_ = malloc_usable_size( E ) / sizeof( DATA_TYPE );
  F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE)); F_LEN_ = malloc_usable_size( F ) / sizeof( DATA_TYPE );
  G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE)); G_LEN_ = malloc_usable_size( G ) / sizeof( DATA_TYPE );
  G_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE)); G_outputFromGpu_LEN_ = malloc_usable_size( G_outputFromGpu ) / sizeof( DATA_TYPE );

  printf("<< Linear Algebra: 3 Matrix Multiplications (E=A.B; F=C.D; G=E.F) >>\n");

  
  #pragma omp target enter data map( alloc: A[:A_LEN_], B[:B_LEN_], C[:C_LEN_], D[:D_LEN_] )
  init_array(A, A_LEN_, B, B_LEN_, C, C_LEN_, D, D_LEN_);

  t_start = rtclock();
  
  #pragma omp target enter data map( to: E[:E_LEN_], F[:F_LEN_], G_outputFromGpu[:G_outputFromGpu_LEN_] )
  
  #pragma omp target update to( A[:A_LEN_], B[:B_LEN_], C[:C_LEN_], D[:D_LEN_] )
  GPU__mm3(A, A_LEN_, B, B_LEN_, C, C_LEN_, D, D_LEN_, E, E_LEN_, F, F_LEN_, G_outputFromGpu, G_outputFromGpu_LEN_);
  
  
  #pragma omp target update from( G_outputFromGpu[:G_outputFromGpu_LEN_] )
  update(G_outputFromGpu, G_outputFromGpu_LEN_);
  
  t_end = rtclock();	

  printf("GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  
  #pragma omp target update from( E[:E_LEN_], F[:F_LEN_] )
  mm3_cpu(A, A_LEN_, B, B_LEN_, C, C_LEN_, D, D_LEN_, E, E_LEN_, F, F_LEN_, G, G_LEN_);
  t_end = rtclock();

  printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(G, G_LEN_, G_outputFromGpu, G_outputFromGpu_LEN_);

  
  #pragma omp target exit data map( delete: A[:A_LEN_] )
  free(A);
  
  #pragma omp target exit data map( delete: B[:B_LEN_] )
  free(B);
  
  #pragma omp target exit data map( delete: C[:C_LEN_] )
  free(C);
  
  #pragma omp target exit data map( delete: D[:D_LEN_] )
  free(D);
  
  #pragma omp target exit data map( delete: E[:E_LEN_] )
  free(E);
  
  #pragma omp target exit data map( delete: F[:F_LEN_] )
  free(F);
  free(G);
  
  #pragma omp target exit data map( delete: G_outputFromGpu[:G_outputFromGpu_LEN_] )
  free(G_outputFromGpu);

  return 0;
}

