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

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>


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

int main();

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1

/* Problem size */
#define N 24576

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1, DATA_TYPE *y2, DATA_TYPE *x1_gpu, DATA_TYPE *x2_gpu) {
    int i, j;

    for (i = 0; i < N; i++) {
        x1[i] = ((DATA_TYPE)i) / N;
        x2[i] = ((DATA_TYPE)i + 1) / N;
        x1_gpu[i] = x1[i];
        x2_gpu[i] = x2[i];
        y1[i] = ((DATA_TYPE)i + 3) / N;
        y2[i] = ((DATA_TYPE)i + 4) / N;
        for (j = 0; j < N; j++) {
            A[i * N + j] = ((DATA_TYPE)i * j) / N;
        }
    }
}

void runMvt(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1, DATA_TYPE *y2) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            x1[i] = x1[i] + a[i * N + j] * y1[j];
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            x2[i] = x2[i] + a[j * N + i] * y2[j];
        }
    }
}

void GPU__runMvt(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1, DATA_TYPE *y2) {
    int i;

    // Note that you must collapse only outer loop to avoid conflicts
    char RST_AI1 = 0;
    RST_AI1 |= !(((void *)(a + 0) > (void *)(x1 + 24575)) || ((void *)(x1 + 0) > (void *)(a + 603979775)));
    RST_AI1 |= !(((void *)(a + 0) > (void *)(x2 + 24575)) || ((void *)(x2 + 0) > (void *)(a + 603979775)));
    RST_AI1 |= !(((void *)(a + 0) > (void *)(y1 + 24575)) || ((void *)(y1 + 0) > (void *)(a + 603979775)));
    RST_AI1 |= !(((void *)(a + 0) > (void *)(y2 + 24575)) || ((void *)(y2 + 0) > (void *)(a + 603979775)));
    RST_AI1 |= !(((void *)(x1 + 0) > (void *)(x2 + 24575)) || ((void *)(x2 + 0) > (void *)(x1 + 24575)));
    RST_AI1 |= !(((void *)(x1 + 0) > (void *)(y1 + 24575)) || ((void *)(y1 + 0) > (void *)(x1 + 24575)));
    RST_AI1 |= !(((void *)(x1 + 0) > (void *)(y2 + 24575)) || ((void *)(y2 + 0) > (void *)(x1 + 24575)));
    RST_AI1 |= !(((void *)(x2 + 0) > (void *)(y1 + 24575)) || ((void *)(y1 + 0) > (void *)(x2 + 24575)));
    RST_AI1 |= !(((void *)(x2 + 0) > (void *)(y2 + 24575)) || ((void *)(y2 + 0) > (void *)(x2 + 24575)));
    RST_AI1 |= !(((void *)(y1 + 0) > (void *)(y2 + 24575)) || ((void *)(y2 + 0) > (void *)(y1 + 24575)));
#pragma omp target data map(to : a [0:603979776], y1 [0:24576], y2 [0:24576]) map(tofrom : x1 [0:24576], x2 [0:24576]) if (!RST_AI1)
    {
#pragma omp target parallel for if (!RST_AI1)
        for (i = 0; i < N; i++) {
            int j;
            for (j = 0; j < N; j++) {
                x1[i] = x1[i] + a[i * N + j] * y1[j];
            }
        }

#pragma omp target parallel for if (!RST_AI1)
        for (i = 0; i < N; i++) {
            int j;
            for (j = 0; j < N; j++) {
                x2[i] = x2[i] + a[j * N + i] * y2[j];
            }
        }
    }
    return;
}

void compareResults(DATA_TYPE *x1, DATA_TYPE *x1_outputFromGpu, DATA_TYPE *x2, DATA_TYPE *x2_outputFromGpu) {
    int i, fail;
    fail = 0;

    for (i = 0; i < N; i++) {
        if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            fail++;
        }

        if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            fail++;
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
           "Percent: %d\n",
           PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

inline void update(DATA_TYPE *x1_outputFromGpu, DATA_TYPE *x2_outputFromGpu) {
    DATA_TYPE cc1 = x1_outputFromGpu[0];
    DATA_TYPE cc2 = x2_outputFromGpu[0];
}

int main() {
    double t_start, t_end;

    DATA_TYPE *a;
    DATA_TYPE *x1;
    DATA_TYPE *x2;
    DATA_TYPE *x1_outputFromGpu;
    DATA_TYPE *x2_outputFromGpu;
    DATA_TYPE *y_1;
    DATA_TYPE *y_2;

    const unsigned long long int threshold = (unsigned long long int)0xFFFFFFFFFFFFFFFF; // 128 * 1024;
    if (threshold > N * N * sizeof(DATA_TYPE)) {
        a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
    } else {
        cudaMallocHost((void **)&a, N * N * sizeof(DATA_TYPE));
    }

    if (threshold > N * sizeof(DATA_TYPE)) {
        x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        x2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        x1_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        x2_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    } else {
        cudaMallocHost((void **)&x1, N * sizeof(DATA_TYPE));
        cudaMallocHost((void **)&x2, N * sizeof(DATA_TYPE));
        cudaMallocHost((void **)&x1_outputFromGpu, N * sizeof(DATA_TYPE));
        cudaMallocHost((void **)&x2_outputFromGpu, N * sizeof(DATA_TYPE));
        cudaMallocHost((void **)&y_1, N * sizeof(DATA_TYPE));
        cudaMallocHost((void **)&y_2, N * sizeof(DATA_TYPE));
    }

    printf("<< Matrix Vector Product and Transpose >>\n");
    printf("N: %d\n", N);

    init_array(a, x1, x2, y_1, y_2, x1_outputFromGpu, x2_outputFromGpu);

    t_start = rtclock();
    GPU__runMvt(a, x1_outputFromGpu, x2_outputFromGpu, y_1, y_2);

    update(x1_outputFromGpu, x2_outputFromGpu);

    t_end = rtclock();
    printf("GPU Runtime(s): %0.6lf\n", t_end - t_start);

    // t_start = rtclock();
    // // run the algorithm on the CPU
    // runMvt(a, x1, x2, y_1, y_2);
    // t_end = rtclock();
    // printf("CPU Runtime: %0.6lfs\n", t_end - t_start);

    // compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);

    if (threshold > N * N * sizeof(DATA_TYPE)) {
        free(a);
    } else {
        cudaFreeHost(a);
    }

    if (threshold > N * sizeof(DATA_TYPE)) {
        free(x1);
        free(x2);
        free(x1_outputFromGpu);
        free(x2_outputFromGpu);
        free(y_1);
        free(y_2);
    } else {
        cudaFreeHost(x1);
        cudaFreeHost(x2);
        cudaFreeHost(x1_outputFromGpu);
        cudaFreeHost(x2_outputFromGpu);
        cudaFreeHost(y_1);
        cudaFreeHost(y_2);
    }

    return 0;
}
