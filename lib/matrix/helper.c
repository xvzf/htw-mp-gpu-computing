#include <stdio.h>
#include <stdint.h>
#include "helper.h"

void print_matrix(uint8_t *a, uintmax_t m, uintmax_t n)
{

    printf("(\n");
    for (uintmax_t i = 0; i < m; i++)
    {
        printf("\t");
        for (uintmax_t j = 0; j < n; j++)
        {
            int idx = i * m + j;
            printf("[m=%ju,n=%ju] %hhu\t", i, j, a[idx]);
        }
        printf("\n");
    }
    printf(")\n");
}

void print_vec(uint8_t *a, uintmax_t n)
{

    printf("(\n");
    for (uintmax_t i = 0; i < n; i++)
    {
        printf("\t[idx=%ju] %hhu\t", i, a[i]);
        printf("\n");
    }
    printf(")\n");
}