#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// #include "common.h"
#define GET_RAND_FP ((float)rand() / ((float)(RAND_MAX) + (float)(1)))

#define MIN(i, j) ((i) < (j) ? (i) : (j))

typedef enum _FUNC_RETURN_CODE { RET_SUCCESS, RET_FAILURE } func_ret_t;

typedef struct __stopwatch_t {
    struct timeval begin;
    struct timeval end;
} stopwatch;

void stopwatch_start(stopwatch *sw) {
    if (sw == NULL)
        return;

    bzero(&sw->begin, sizeof(struct timeval));
    bzero(&sw->end, sizeof(struct timeval));

    gettimeofday(&sw->begin, NULL);
}

void stopwatch_stop(stopwatch *sw) {
    if (sw == NULL)
        return;

    gettimeofday(&sw->end, NULL);
}

double get_interval_by_sec(stopwatch *sw) {
    if (sw == NULL)
        return 0;
    return ((double)(sw->end.tv_sec - sw->begin.tv_sec) + (double)(sw->end.tv_usec - sw->begin.tv_usec) / 1000000);
}

int get_interval_by_usec(stopwatch *sw) {
    if (sw == NULL)
        return 0;
    return ((sw->end.tv_sec - sw->begin.tv_sec) * 1000000 + (sw->end.tv_usec - sw->begin.tv_usec));
}

int main(int argc, char *argv[]);
extern char *optarg;
extern int optind, opterr, optopt;

func_ret_t create_matrix_from_file(float **mp, const char *filename, int *size_p) {
    int i, j, size;
    float *m;
    FILE *fp = NULL;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        return RET_FAILURE;
    }

    fscanf(fp, "%d\n", &size);

    m = (float *)malloc(sizeof(float) * size * size);
    if (m == NULL) {
        fclose(fp);
        return RET_FAILURE;
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            fscanf(fp, "%f ", m + i * size + j);
        }
    }

    fclose(fp);

    *size_p = size;
    *mp = m;

    return RET_SUCCESS;
}

func_ret_t create_matrix_from_random(float **mp, int size) {
    float *l, *u, *m;
    int i, j, k;

    srand(time(NULL));

    l = (float *)malloc(size * size * sizeof(float));
    if (l == NULL)
        return RET_FAILURE;

    u = (float *)malloc(size * size * sizeof(float));
    if (u == NULL) {
        free(l);
        return RET_FAILURE;
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            if (i > j) {
                l[i * size + j] = GET_RAND_FP;
            } else if (i == j) {
                l[i * size + j] = 1;
            } else {
                l[i * size + j] = 0;
            }
        }
    }

    for (j = 0; j < size; j++) {
        for (i = 0; i < size; i++) {
            if (i > j) {
                u[j * size + i] = 0;
            } else {
                u[j * size + i] = GET_RAND_FP;
            }
        }
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            for (k = 0; k <= MIN(i, j); k++)
                m[i * size + j] = l[i * size + k] * u[j * size + k];
        }
    }

    free(l);
    free(u);

    *mp = m;

    return RET_SUCCESS;
}

void matrix_multiply(float *inputa, float *inputb, float *output, int size) {
    int i, j, k;

    for (i = 0; i < size; i++)
        for (k = 0; k < size; k++)
            for (j = 0; j < size; j++)
                output[i * size + j] = inputa[i * size + k] * inputb[k * size + j];
}

func_ret_t lud_verify(float *m, float *lu, int matrix_dim) {
    int i, j, k;
    float *tmp = (float *)malloc(matrix_dim * matrix_dim * sizeof(float));

    for (i = 0; i < matrix_dim; i++)
        for (j = 0; j < matrix_dim; j++) {
            float sum = 0;
            float l, u;
            for (k = 0; k <= MIN(i, j); k++) {
                if (i == k)
                    l = 1;
                else
                    l = lu[i * matrix_dim + k];
                u = lu[k * matrix_dim + j];
                sum += l * u;
            }
            tmp[i * matrix_dim + j] = sum;
        }
    /* printf(">>>>>LU<<<<<<<\n"); */
    /* for (i=0; i<matrix_dim; i++){ */
    /*   for (j=0; j<matrix_dim;j++){ */
    /*       printf("%f ", lu[i*matrix_dim+j]); */
    /*   } */
    /*   printf("\n"); */
    /* } */
    /* printf(">>>>>result<<<<<<<\n"); */
    /* for (i=0; i<matrix_dim; i++){ */
    /*   for (j=0; j<matrix_dim;j++){ */
    /*       printf("%f ", tmp[i*matrix_dim+j]); */
    /*   } */
    /*   printf("\n"); */
    /* } */
    /* printf(">>>>>input<<<<<<<\n"); */
    /* for (i=0; i<matrix_dim; i++){ */
    /*   for (j=0; j<matrix_dim;j++){ */
    /*       printf("%f ", m[i*matrix_dim+j]); */
    /*   } */
    /*   printf("\n"); */
    /* } */

    for (i = 0; i < matrix_dim; i++) {
        for (j = 0; j < matrix_dim; j++) {
            if (fabs(m[i * matrix_dim + j] - tmp[i * matrix_dim + j]) > 0.0001)
                printf("dismatch at (%d, %d): (o)%f (n)%f\n", i, j, m[i * matrix_dim + j], tmp[i * matrix_dim + j]);
        }
    }
    free(tmp);

    return RET_SUCCESS;
}

void matrix_duplicate(float *src, float **dst, int matrix_dim) {
    int s = matrix_dim * matrix_dim * sizeof(float);
    float *p = (float *)malloc(s);
    memcpy(p, src, s);
    *dst = p;
}

void print_matrix(float *m, int matrix_dim) {
    int i, j;
    for (i = 0; i < matrix_dim; i++) {
        for (j = 0; j < matrix_dim; j++)
            printf("%f ", m[i * matrix_dim + j]);
        printf("\n");
    }
}

// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

func_ret_t create_matrix(float **mp, int size) {
    float *m;
    int i, j;
    float lamda = -0.001;
    float coe[2 * size - 1];
    float coe_i = 0.0;

    for (i = 0; i < size; i++) {
        coe_i = 10 * exp(lamda * i);
        j = size - 1 + i;
        coe[j] = coe_i;
        j = size - 1 - i;
        coe[j] = coe_i;
    }

    m = (float *)malloc(sizeof(float) * size * size);
    if (m == NULL) {
        return RET_FAILURE;
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            m[i * size + j] = coe[size - 1 - i + j];
        }
    }

    *mp = m;

    return RET_SUCCESS;
}

///////////////////////////////////////////////////////
static int do_verify = 0;
int omp_num_threads = 36;

#define BS 16

#define AA(_i, _j) a[offset * size + _i * size + _j + offset]
#define BB(_i, _j) a[_i * size + _j]

#ifdef OMP_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

void lud_diagonal_omp(float *a, int size, int offset) {
    int i, j, k;
    for (i = 0; i < BS; i++) {

        for (j = i; j < BS; j++) {
            for (k = 0; k < i; k++) {
                AA(i, j) = AA(i, j) - AA(i, k) * AA(k, j);
            }
        }

        float temp = 1.f / AA(i, i);
        for (j = i + 1; j < BS; j++) {
            for (k = 0; k < i; k++) {
                AA(j, i) = AA(j, i) - AA(j, k) * AA(k, i);
            }
            AA(j, i) = AA(j, i) * temp;
        }
    }
}

#ifdef OMP_OFFLOAD
#pragma offload_attribute(pop)
#endif

// implements block LU factorization
void lud_omp(float *a, int size) {
    int offset, chunk_idx, size_inter, chunks_in_inter_row, chunks_per_inter;

#ifdef OMP_OFFLOAD
#pragma omp target map(to : size) map(a [0:size * size])
#endif

#ifdef OMP_OFFLOAD
    {
        // omp_set_num_threads(224);
#else
    // printf("running OMP on host\n");
    // omp_set_num_threads(omp_num_threads);
#endif
        for (offset = 0; offset < size - BS; offset += BS) {
            // lu factorization of left-top corner block diagonal matrix
            //
            lud_diagonal_omp(a, size, offset);

            size_inter = size - offset - BS;
            chunks_in_inter_row = size_inter / BS;

// calculate perimeter block matrices
//
#pragma omp target data map(tofrom : size, a [0:size * size])
            {
#pragma omp target teams distribute parallel for private(chunk_idx) shared(size, chunks_per_inter, chunks_in_inter_row, offset, a)
                for (chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++) {
                    int i, j, k, i_global, j_global, i_here, j_here;
                    float sum;
                    float temp[BS * BS] __attribute__((aligned(64)));

                    for (i = 0; i < BS; i++) {
#pragma omp simd
                        for (j = 0; j < BS; j++) {
                            temp[i * BS + j] = a[size * (i + offset) + offset + j];
                        }
                    }
                    i_global = offset;
                    j_global = offset;

                    // processing top perimeter
                    //
                    j_global += BS * (chunk_idx + 1);
                    for (j = 0; j < BS; j++) {
                        for (i = 0; i < BS; i++) {
                            sum = 0.f;
                            for (k = 0; k < i; k++) {
                                sum += temp[BS * i + k] * BB((i_global + k), (j_global + j));
                            }
                            i_here = i_global + i;
                            j_here = j_global + j;
                            BB(i_here, j_here) = BB(i_here, j_here) - sum;
                        }
                    }

                    // processing left perimeter
                    //
                    j_global = offset;
                    i_global += BS * (chunk_idx + 1);
                    for (i = 0; i < BS; i++) {
                        for (j = 0; j < BS; j++) {
                            sum = 0.f;
                            for (k = 0; k < j; k++) {
                                sum += BB((i_global + i), (j_global + k)) * temp[BS * k + j];
                            }
                            i_here = i_global + i;
                            j_here = j_global + j;
                            a[size * i_here + j_here] = (a[size * i_here + j_here] - sum) / a[size * (offset + j) + offset + j];
                        }
                    }
                }
            }

            // update interior block matrices
            //
            chunks_per_inter = chunks_in_inter_row * chunks_in_inter_row;
#pragma omp target data map(tofrom : size, a [0:size * size])
            {
#pragma omp target teams distribute parallel for private(chunk_idx)                                                     \
    shared(size, chunks_per_inter, chunks_in_inter_row, offset, a)
                for (chunk_idx = 0; chunk_idx < chunks_per_inter; chunk_idx++) {
                    int i, j, k, i_global, j_global;
                    float temp_top[BS * BS] __attribute__((aligned(64)));
                    float temp_left[BS * BS] __attribute__((aligned(64)));
                    float sum[BS] __attribute__((aligned(64))) = {0.f};

                    i_global = offset + BS * (1 + chunk_idx / chunks_in_inter_row);
                    j_global = offset + BS * (1 + chunk_idx % chunks_in_inter_row);

                    for (i = 0; i < BS; i++) {
#pragma omp simd
                        for (j = 0; j < BS; j++) {
                            temp_top[i * BS + j] = a[size * (i + offset) + j + j_global];
                            temp_left[i * BS + j] = a[size * (i + i_global) + offset + j];
                        }
                    }

                    for (i = 0; i < BS; i++) {
                        for (k = 0; k < BS; k++) {
#pragma omp simd
                            for (j = 0; j < BS; j++) {
                                sum[j] += temp_left[BS * i + k] * temp_top[BS * k + j];
                            }
                        }
#pragma omp simd
                        for (j = 0; j < BS; j++) {
                            BB((i + i_global), (j + j_global)) -= sum[j];
                            sum[j] = 0.f;
                        }
                    }
                }
            }
        }

        lud_diagonal_omp(a, size, offset);
#ifdef OMP_OFFLOAD
    }
#endif
}

static struct option long_options[] = {
    /* name, has_arg, flag, val */
    {"input", 1, NULL, 'i'},
    {"size", 1, NULL, 's'},
    {"verify", 0, NULL, 'v'},
    {0, 0, 0, 0}};

int main(int argc, char *argv[]) {
    int matrix_dim = 32; /* default size */
    int opt, option_index = 0;
    func_ret_t ret;
    const char *input_file = NULL;
    float *m, *mm;
    stopwatch sw;

    while ((opt = getopt_long(argc, argv, "::vs:n:i:", long_options, &option_index)) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'v':
            do_verify = 1;
            break;
        case 'n':
            omp_num_threads = atoi(optarg);
            break;
        case 's':
            matrix_dim = atoi(optarg);
            printf("Generate input matrix internally, size =%d\n", matrix_dim);
            // fprintf(stderr, "Currently not supported, use -i instead\n");
            // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
            // exit(EXIT_FAILURE);
            break;
        case '?':
            fprintf(stderr, "invalid option\n");
            break;
        case ':':
            fprintf(stderr, "missing argument\n");
            break;
        default:
            fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if ((optind < argc) || (optind == 1)) {
        fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (input_file) {
        printf("Reading matrix from file %s\n", input_file);
        ret = create_matrix_from_file(&m, input_file, &matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix from file %s\n", input_file);
            exit(EXIT_FAILURE);
        }
    } else if (matrix_dim) {
        printf("Creating matrix internally size=%d\n", matrix_dim);
        ret = create_matrix(&m, matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
            exit(EXIT_FAILURE);
        }
    }

    else {
        printf("No input file specified!\n");
        exit(EXIT_FAILURE);
    }

    if (do_verify) {
        printf("Before LUD\n");
        /* print_matrix(m, matrix_dim); */
        matrix_duplicate(m, &mm, matrix_dim);
    }

    stopwatch_start(&sw);
    lud_omp(m, matrix_dim);
    stopwatch_stop(&sw);
    printf("Time consumed(ms): %lf\n", 1000 * get_interval_by_sec(&sw));

    if (do_verify) {
        printf("After LUD\n");
        /* print_matrix(m, matrix_dim); */
        printf(">>>Verify<<<<\n");
        lud_verify(mm, m, matrix_dim);
        free(mm);
    }

    free(m);

    return EXIT_SUCCESS;
} /* ----------  end of function main  ---------- */
