#include <stdio.h>
#include <stdint.h>
#include "mat_helper.h"

void print_matrix(uint8_t *a, uint64_t m, uint64_t n)
{

    printf("(\n");
    for (uint64_t i = 0; i < m; i++)
    {
        printf("\t");
        for (uint64_t j = 0; j < n; j++)
        {
            int idx = i * m + j;
            printf("[m=%ld,n=%ld] %0.0f\t", i, j, a[idx]);
        }
        printf("\n");
    }
    printf(")\n");
}

void print_vec(uint8_t *a, uint64_t n)
{

    printf("(\n");
    for (uint64_t i = 0; i < n; i++)
    {
        printf("\t[idx=%ld] %0.0f\t", i, a[i]);
        printf("\n");
    }
    printf(")\n");
}