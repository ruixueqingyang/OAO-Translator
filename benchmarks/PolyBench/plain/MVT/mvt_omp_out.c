#include <malloc.h>
#include "RunTime.h"
STATE_CONSTR StConstrTarget;
/**
 * mvt.c: This file was adapted from PolyBench/GPU 1.0 test suite
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

#define GPU_DEVICE 1


#ifdef _DEBUG_1
/* Problem size */
#define N 24576

#elif _DEBUG_2
/* Problem size */
#define N 12288

#else
/* Problem size */
#define N 4096
#endif



/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2, DATA_TYPE* x1_gpu, DATA_TYPE* x2_gpu)
{
  int i, j;

  for (i = 0; i < N; i++)
    {
      x1[i] = ((DATA_TYPE) i) / N;
      x2[i] = ((DATA_TYPE) i + 1) / N;
      x1_gpu[i] = x1[i]; 
      x2_gpu[i] = x2[i];
      y1[i] = ((DATA_TYPE) i + 3) / N;
      y2[i] = ((DATA_TYPE) i + 4) / N;
      for (j = 0; j < N; j++)
	{
	  A[i*N + j] = ((DATA_TYPE) i*j) / N;
	}
    }
}

void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
  int i, j;
	
  for (i=0; i<N; i++) 
    {
      for (j=0; j<N; j++) 
	{
	  x1[i] = x1[i] + a[i*N + j] * y1[j];
	}
    }
  
  for (i=0; i<N; i++) 
    {
      for (j=0; j<N; j++) 
	{
	  x2[i] = x2[i] + a[j*N + i] * y2[j];
	}
    }
}

void GPU__runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
  int i;
  
  //Note that you must collapse only outer loop to avoid conflicts
  #pragma omp target teams distribute parallel for map(tofrom: i)
  for (i=0; i<N; i++) 
    {
      int j;
      for (j=0; j<N; j++) 
	{
	  x1[i] = x1[i] + a[i*N + j] * y1[j];
	}
    }
    OAOStTrans( x1, StConstrTarget.init(5, 4) );
    
	
  #pragma omp target teams distribute parallel for map(tofrom: i)
  for (i=0; i<N; i++) 
    {
      int j;
      for (j=0; j<N; j++) 
	{
	  x2[i] = x2[i] + a[j*N + i] * y2[j];
	}
    }
    OAOStTrans( x2, StConstrTarget.init(5, 4) );
    
  return;
}

void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
  int i, fail;
  fail = 0;
	
  for (i=0; i<N; i++) 
    {
      if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
	{
	  fail++;
	}
      
      if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
	{
	  fail++;
	}
    }
	
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

inline void update(DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2_outputFromGpu)
{
  DATA_TYPE cc1 = x1_outputFromGpu[0];
  DATA_TYPE cc2 = x2_outputFromGpu[0];
}

int main()
{
  double t_start, t_end;

  DATA_TYPE* a;
  DATA_TYPE* x1;
  DATA_TYPE* x2;
  DATA_TYPE* x1_outputFromGpu;
  DATA_TYPE* x2_outputFromGpu;
  DATA_TYPE* y_1;
  DATA_TYPE* y_2;

  a = (DATA_TYPE*)OAOMalloc(N*N*sizeof(DATA_TYPE));
  x1 = (DATA_TYPE*)OAOMalloc(N*sizeof(DATA_TYPE));
  x2 = (DATA_TYPE*)OAOMalloc(N*sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE*)OAOMalloc(N*sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE*)OAOMalloc(N*sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE*)OAOMalloc(N*sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE*)OAOMalloc(N*sizeof(DATA_TYPE));

  printf("<< Matrix Vector Product and Transpose >>\n");
  printf("Data size N: %d\n", N);

  init_array(a, x1, x2, y_1, y_2, x1_outputFromGpu, x2_outputFromGpu);
	
  t_start = rtclock();
  
  OAODataTrans( a, StConstrTarget.init(7, 5) );
  OAODataTrans( x1_outputFromGpu, StConstrTarget.init(7, 5) );
  OAODataTrans( x2_outputFromGpu, StConstrTarget.init(7, 5) );
  OAODataTrans( y_1, StConstrTarget.init(7, 5) );
  OAODataTrans( y_2, StConstrTarget.init(7, 5) );
  GPU__runMvt(a, x1_outputFromGpu, x2_outputFromGpu, y_1, y_2);
  
  
  OAODataTrans( x1_outputFromGpu, StConstrTarget.init(7, 2) );
  OAODataTrans( x2_outputFromGpu, StConstrTarget.init(7, 2) );
  update(x1_outputFromGpu, x2_outputFromGpu); 
  
  t_end = rtclock();
  printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);
  
  // t_start = rtclock();
  // //run the algorithm on the CPU
  // runMvt(a, x1, x2, y_1, y_2);
  // t_end = rtclock();
  // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);
  
  // compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);
  
  
  OAODataTrans( a, StConstrTarget.init(2, 2) );
  OAOFree(a);
  OAOFree(x1);
  OAOFree(x2);
  
  OAODataTrans( x1_outputFromGpu, StConstrTarget.init(2, 2) );
  OAOFree(x1_outputFromGpu);
  
  OAODataTrans( x2_outputFromGpu, StConstrTarget.init(2, 2) );
  OAOFree(x2_outputFromGpu);
  
  OAODataTrans( y_1, StConstrTarget.init(2, 2) );
  OAOFree(y_1);
  
  OAODataTrans( y_2, StConstrTarget.init(2, 2) );
  OAOFree(y_2);

  OAOExpenseTime();
  //DataTransInfo();
  return 0;

OAODataTrans( a, StConstrTarget.init(2, 2) );
OAODataTrans( x1_outputFromGpu, StConstrTarget.init(2, 2) );
OAODataTrans( x2_outputFromGpu, StConstrTarget.init(2, 2) );
OAODataTrans( y_1, StConstrTarget.init(2, 2) );
OAODataTrans( y_2, StConstrTarget.init(2, 2) );
}

