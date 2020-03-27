#include <malloc.h>
#include "RunTime.h"
STATE_CONSTR StConstrTarget;
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>

#define N 10

int main();

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

struct float3 ff_fc_momentum_x;


void GPU__correlation(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev,int xx[10])
{
  int i, j;
  int cc[10] = {0,0,0,0,0,0,0,0,0,0}; OAOArrayInfo( (void*)(cc), sizeof(int [10]), sizeof(int ) );
// Determine mean of column vectors of input data matrix
OAODataTrans( cc, StConstrTarget.init(7, 5) );

#pragma #pragma omp target teams distribute parallel for 
  for (j = 0; j < N; j++)
  {
    mean[j] = xx[j] + cc[j];
    for (i = 0; i < N; i++)
    {
      data[i] = mean[j];
    }
  }
  OAOStTrans( data, StConstrTarget.init(5, 4) );
  OAOStTrans( mean, StConstrTarget.init(5, 4) );
  
  

  
  OAODataTrans( data, StConstrTarget.init(7, 2) );
  OAODataTrans( mean, StConstrTarget.init(7, 2) );
  for (j = 0; j < N; j++)
  {
    stddev[j] = 0.0;
    //printf("xx = %d \n", xx[j]);
    for (i = 0; i < N; i++)
    {
      stddev[j] += data[i] + mean[j];
    }
  }

  
OAODeleteArray( (void*)(cc) );
  return;
}


int main()
{
  double t_start, t_end;

  DATA_TYPE *data;
  DATA_TYPE *mean;
  DATA_TYPE *stddev;

  int xx[10] = {0,1,2,3,4,5,6,7,8,9}; OAOArrayInfo( (void*)(xx), sizeof(int [10]), sizeof(int ) ); //静态数组
  data = (DATA_TYPE *)OAOMalloc(N * sizeof(DATA_TYPE));
  mean = (DATA_TYPE *)OAOMalloc(N * sizeof(DATA_TYPE));
  stddev = (DATA_TYPE *)OAOMalloc(N * sizeof(DATA_TYPE));
  
  int m;
  for (m = 0; m < N; m++)
  {
    //mean[i] = i;
    data[m] = m;
  }
  OAOStTrans( data, StConstrTarget.init(3, 2) );
  


  OAODataTrans( data, StConstrTarget.init(7, 5) );
  OAODataTrans( mean, StConstrTarget.init(7, 5) );
  OAODataTrans( xx, StConstrTarget.init(7, 5) );
    GPU__correlation(data, mean,stddev,xx);
  
  
  OAODataTrans( mean, StConstrTarget.init(7, 2) );
  for (int i = 0; i < N; i++)
  {
    printf("mean: %f\n", mean[i]);
  }
  
  
  OAODataTrans( data, StConstrTarget.init(7, 2) );
  for (int i = 0; i < N; i++)
  {
    printf("data: %f\n", data[i]);
  }


  OAODataTrans( data, StConstrTarget.init(2, 2) );
    OAOFree(data);

  OAODataTrans( mean, StConstrTarget.init(2, 2) );
    OAOFree(mean);
  OAOFree(stddev);

  
OAODeleteArray( (void*)(xx) );
  return 0;
}

