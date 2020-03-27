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


void GPU__correlation(DATA_TYPE *data, DATA_TYPE *mean, DATA_TYPE *stddev,int xx[10])
{
  int i, j;
  int cc[10] = {0,0,0,0,0,0,0,0,0,0};
// Determine mean of column vectors of input data matrix
#pragma omp parallel for
  for (j = 0; j < N; j++)
  {
    mean[j] = xx[j] + cc[j];
    for (i = 0; i < N; i++)
    {
      data[i] = mean[j];
    }
  }
  

  for (j = 0; j < N; j++)
  {
    stddev[j] = 0.0;
    //printf("xx = %d \n", xx[j]);
    for (i = 0; i < N; i++)
    {
      stddev[j] += data[i] + mean[j];
    }
  }

  return;
}


int main()
{
  double t_start, t_end;

  DATA_TYPE *data;
  DATA_TYPE *mean;
  DATA_TYPE *stddev;

  int xx[10] = {0,1,2,3,4,5,6,7,8,9}; //静态数组
  data = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  mean = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  stddev = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  
  int m;
  for (m = 0; m < N; m++)
  {
    //mean[i] = i;
    data[m] = m;
  }

  GPU__correlation(data, mean,stddev,xx);
  
  for (int i = 0; i < N; i++)
  {
    printf("mean: %f\n", mean[i]);
  }
  
  for (int i = 0; i < N; i++)
  {
    printf("data: %f\n", data[i]);
  }

  free(data);
  free(mean);
  free(stddev);

  return 0;
}

