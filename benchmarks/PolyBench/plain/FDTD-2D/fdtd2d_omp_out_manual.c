#include <malloc.h>
/**
 * fdtd2d.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
 */

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>

// polybenchUtilFuncts.h
// Scott Grauer-Gray (sgrauerg@gmail.com)
// Functions used across hmpp codes

#ifndef POLYBENCH_UTIL_FUNCTS_H
#define POLYBENCH_UTIL_FUNCTS_H

// define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a) {
    if (a < 0) {
        return (a * -1);
    } else {
        return a;
    }
}

float percentDiff(double val1, double val2) {
    if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)) {
        return 0.0f;
    }

    else {
        return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
    }
}

#endif // POLYBENCH_UTIL_FUNCTS_H

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU 1


#ifdef _DEBUG_1
/* Problem size */
#define tmax 1500
#define NX 8192
#define NY 8192

#elif _DEBUG_2
/* Problem size */
#define tmax 1000
#define NX 4096
#define NY 4096

#else
/* Problem size */
#define tmax 500
#define NX 2048
#define NY 2048
#endif


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz) {
    int i, j;

    for (i = 0; i < tmax; i++) {
        _fict_[i] = (DATA_TYPE)i;
    }

    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
    }
}

void init_array_hz(DATA_TYPE *hz) {
    int i, j;

    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
    }
}

void compareResults(DATA_TYPE *hz1, DATA_TYPE *hz2) {
    int i, j, fail;
    fail = 0;

    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            if (percentDiff(hz1[i * NY + j], hz2[i * NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD,
           fail);
}

void runFdtd(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz) {
    int t, i, j;

    for (t = 0; t < tmax; t++) {
        for (j = 0; j < NY; j++) {
            ey[0 * NY + j] = _fict_[t];
        }

        for (i = 1; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }

        for (i = 0; i < NX; i++) {
            for (j = 1; j < NY; j++) {
                ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }

        for (i = 0; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                                                         ey[(i + 1) * NY + j] - ey[i * NY + j]);
            }
        }
    }
}

int main();

void runFdtd_OMP(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz) {
    int t, i, j;

    int Length1 = tmax;
    int Length2 = NX * (NY + 1);
    int Length3 = (NX + 1) * NY;
    int Length4 = NX * NY;

    for (t = 0; t < tmax; t++) {
#pragma omp target data map( tofrom: _fict_[:Length1], ey[:Length3] )
{
#pragma omp target teams distribute parallel for private(j) collapse(1)
        for (j = 0; j < NY; j++) {
            ey[0 * NY + j] = _fict_[t];
        }
}

#pragma omp target data map( tofrom: ey[:Length3], hz[:Length4] )
{
#pragma omp target teams distribute parallel for private(i, j) collapse(2)
        for (i = 1; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }
}

#pragma omp target data map( tofrom: ex[:Length2], hz[:Length4] )
{
#pragma omp target teams distribute parallel for private(i, j) collapse(2)
        for (i = 0; i < NX; i++) {
            for (j = 1; j < NY; j++) {
                ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }
}

#pragma omp target data map( tofrom: ex[:Length2], ey[:Length3], hz[:Length4] )
{
#pragma omp target teams distribute parallel for private(i, j) collapse(2)
        for (i = 0; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] +
                                                         ey[(i + 1) * NY + j] - ey[i * NY + j]);
            }
        }
}
    }

}

inline void foo(DATA_TYPE *hz){
    DATA_TYPE tmp = hz[0];
    return;
}

int main() {
    double t_start, t_end;
    double s1, s2, e1, e2;

    DATA_TYPE *_fict_;
    DATA_TYPE *ex;
    DATA_TYPE *ey;
    DATA_TYPE *hz;
    DATA_TYPE *hz_outputFromGpu;

    const unsigned long long int threshold = (unsigned long long int)128 * 1024;
    if(threshold > tmax*sizeof(DATA_TYPE)){
        _fict_ = (DATA_TYPE *)malloc(tmax * sizeof(DATA_TYPE));
    }else{
        cudaMallocHost((void**)&_fict_, tmax*sizeof(DATA_TYPE));
    }

    if(threshold > NX * (NY + 1)*sizeof(DATA_TYPE)){
        ex = (DATA_TYPE *)malloc(NX * (NY + 1) * sizeof(DATA_TYPE));
    }else{
        cudaMallocHost((void**)&ex, NX * (NY + 1)*sizeof(DATA_TYPE));
    }

    if(threshold > (NX + 1) * NY * sizeof(DATA_TYPE)){
        ey = (DATA_TYPE *)malloc((NX + 1) * NY * sizeof(DATA_TYPE));
    }else{
        cudaMallocHost((void**)&ey, (NX + 1) * NY * sizeof(DATA_TYPE));
    }

    if(threshold > NX * NY * sizeof(DATA_TYPE)){
        hz = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
        hz_outputFromGpu = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
    }else{
        cudaMallocHost((void**)&hz, NX * NY * sizeof(DATA_TYPE));
        cudaMallocHost((void**)&hz_outputFromGpu, NX * NY * sizeof(DATA_TYPE));
    }

    printf("<< 2-D Finite Different Time Domain Kernel >>\n");
    printf("NX: %d\n", NX);
    printf("NY: %d\n", NY);

    // init_arrays(_fict_, ex, ey, hz);

    // t_start = rtclock();
    // runFdtd(_fict_, ex, ey, hz);
    // t_end = rtclock();

    // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

    init_arrays(_fict_, ex, ey, hz_outputFromGpu);
    // init_array_hz(hz_outputFromGpu);

    t_start = rtclock();

    runFdtd_OMP(_fict_, ex, ey, hz_outputFromGpu);

    foo(hz_outputFromGpu);
    t_end = rtclock();

    printf("GPU Runtime(s): %0.6lfs\n", t_end - t_start);

    // compareResults(hz, hz_outputFromGpu);
  
    // printf("Host to Device number of times is: %d\n", 4);
    // printf("Host to Device number of byte is: %ld\n", (Length1+Length2+Length3+Length4) * sizeof(DATA_TYPE));
    // printf("Host to Device time is: %0.6lf\n", e1 - s1);
  
    // printf("Device to Host number of times is: %d\n", 4);
    // printf("Device to Host number of byte is: %ld\n", (Length1+Length2+Length3+Length4) * sizeof(DATA_TYPE));
    // printf("Host to Device time is: %0.6lf\n", e2 - s2);
    if(threshold > tmax*sizeof(DATA_TYPE)){
        free(_fict_);
    }else{
        cudaFreeHost(_fict_);
    }

    if(threshold > NX * (NY + 1)*sizeof(DATA_TYPE)){
        free(ex);
    }else{
        cudaFreeHost(ex);
    }

    if(threshold > (NX + 1) * NY * sizeof(DATA_TYPE)){
        free(ey);
    }else{
        cudaFreeHost(ey);
    }

    if(threshold > NX * NY * sizeof(DATA_TYPE)){
        free(hz);
        free(hz_outputFromGpu);
    }else{
        cudaFreeHost(hz);
        cudaFreeHost(hz_outputFromGpu);
    }
    
    return 0;
}