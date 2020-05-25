//
// Created by developer on 5/24/20.
//

#include <malloc.h>
#include <algorithm>

#include "middle.h"
#include "saxpy.cuh"

void mid(int value, float a) {
    int N = 1 << value;
    float *x, *y;

    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    saxpy(N, a, x, y);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = std::max(maxError, std::abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);

    free(x);
    free(y);
}