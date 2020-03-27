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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

/* Problem size. */
#define NX 8192
#define NY 8192

#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, int x_LEN_, DATA_TYPE *A, int A_LEN_)
{
  int i, j;

  for (i = 0; i < NX; i++)
    {
      x[i] = i * M_PI;
      for (j = 0; j < NY; j++)
	{
	  A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
	}
    }
}

void compareResults(DATA_TYPE *z, int z_LEN_, DATA_TYPE *z_outputFromGpu, int z_outputFromGpu_LEN_)
{
  int i, fail;
  fail = 0;

  for (i=0; i<NY; i++)
    {
      if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
	{
	  fail++;
	}		
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void atax_cpu(DATA_TYPE* A, int A_LEN_, DATA_TYPE* x, int x_LEN_, DATA_TYPE* y, int y_LEN_, DATA_TYPE* tmp, int tmp_LEN_)
{
  int i,j;
	
  for (i= 0; i < NY; i++)
    {
      y[i] = 0;
    }
  
  for (i = 0; i < NX; i++)
    {
      tmp[i] = 0;
      
      for (j = 0; j < NY; j++)
	{
	  tmp[i] = tmp[i] + A[i*NY + j] * x[j];
	}
      
      for (j = 0; j < NY; j++)
	{
	  y[j] = y[j] + A[i*NY + j] * tmp[i];
	}
    }
}

void GPU__atax(DATA_TYPE* A, int A_LEN_, DATA_TYPE* x, int x_LEN_, DATA_TYPE* y, int y_LEN_, DATA_TYPE* tmp, int tmp_LEN_)
{
  int i,j;
	
  for (i= 0; i < NY; i++)
    {
      y[i] = 0;
    }
  
  #pragma omp target teams distribute parallel for 
  for (i = 0; i < NX; i++)
    {
      tmp[i] = 0;
      for (j = 0; j < NY; j++)
	{
	  tmp[i] = tmp[i] + A[i*NY + j] * x[j];
	}
    }

  //Note that the Loop has been reversed
  #pragma omp target update to( y[:y_LEN_] )
  
  #pragma omp target teams distribute parallel for 
  for (j = 0; j < NY; j++)
    for (i = 0; i < NX; i++){
      {
	y[j] = y[j] + A[i*NY + j] * tmp[i];
      }
    }
  return;
}

inline void update(DATA_TYPE* y_outputFromGpu, int y_outputFromGpu_LEN_)
{
  DATA_TYPE yy = y_outputFromGpu[0];
}

//int main(int argc, char** argv)
int main()
{
  double t_start, t_end;

  DATA_TYPE* A; int A_LEN_;
  DATA_TYPE* x; int x_LEN_;
  DATA_TYPE* y; int y_LEN_;
  DATA_TYPE* y_outputFromGpu; int y_outputFromGpu_LEN_;
  DATA_TYPE* tmp; int tmp_LEN_;

  A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE)); A_LEN_ = malloc_usable_size( A ) / sizeof( DATA_TYPE );
  x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE)); x_LEN_ = malloc_usable_size( x ) / sizeof( DATA_TYPE );
  y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE)); y_LEN_ = malloc_usable_size( y ) / sizeof( DATA_TYPE );
  y_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE)); y_outputFromGpu_LEN_ = malloc_usable_size( y_outputFromGpu ) / sizeof( DATA_TYPE );
  tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE)); tmp_LEN_ = malloc_usable_size( tmp ) / sizeof( DATA_TYPE );

  printf( "<< Matrix Transpose and Vector Multiplication >>\n");

  
  #pragma omp target enter data map( alloc: A[:A_LEN_], x[:x_LEN_] )
  init_array(x, x_LEN_, A, A_LEN_);

  t_start = rtclock();
  
  #pragma omp target enter data map( to: y_outputFromGpu[:y_outputFromGpu_LEN_], tmp[:tmp_LEN_] )
  
  #pragma omp target update to( A[:A_LEN_], x[:x_LEN_] )
  GPU__atax(A, A_LEN_, x, x_LEN_, y_outputFromGpu, y_outputFromGpu_LEN_, tmp, tmp_LEN_);
  
  
  #pragma omp target update from( y_outputFromGpu[:y_outputFromGpu_LEN_] )
  update(y_outputFromGpu, y_outputFromGpu_LEN_);
  
  t_end = rtclock();
  printf("GPU Runtime: %0.6lfs\n", t_end - t_start);
	
  t_start = rtclock();
  
  #pragma omp target update from( tmp[:tmp_LEN_] )
  atax_cpu(A, A_LEN_, x, x_LEN_, y, y_LEN_, tmp, tmp_LEN_);
  t_end = rtclock();
  printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_LEN_, y_outputFromGpu, y_outputFromGpu_LEN_);

  
  #pragma omp target exit data map( delete: A[:A_LEN_] )
  free(A);
  
  #pragma omp target exit data map( delete: x[:x_LEN_] )
  free(x);
  free(y);
  
  #pragma omp target exit data map( delete: y_outputFromGpu[:y_outputFromGpu_LEN_] )
  free(y_outputFromGpu);
  
  #pragma omp target exit data map( delete: tmp[:tmp_LEN_] )
  free(tmp);

  return 0;
}

