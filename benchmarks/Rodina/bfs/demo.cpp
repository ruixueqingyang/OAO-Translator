#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define N 65536

struct A{
    int length;
    int width;
};

int main(){
    int size = N;
    int error = 0;

    A* node = (A*) malloc(sizeof(A)*size);
    int* result = (int*)malloc(N*sizeof(int));

    for(int i = 0; i < size ; i++){
        node[i].length = i;
        node[i].width = i;
        result[i] = i;
    }

    printf("Start!\n");

//#pragma omp target data map(to: size, node[0:size])   
#pragma omp parallel for
    for(int i = 0; i < size; i++){
        result[i] = node[i].length + node[i].width;
    }

    for(int i = 0; i < size; i++){
        if( (result[i] - node[i].length - node[i].width) != 0 ){
            printf("result=%3d, node=%3d, node=%3d\n", result[i], node[i].length, node[i].width);
            error ++;
        }
    }

    printf("  errors: %d\n", error);
    printf("  End!\n");
	return 0;
}
