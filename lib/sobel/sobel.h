#ifndef __SOBEL_H__
#define __SOBEL_H__

#include <stdint.h>
#include "lib/img/types.h"

// Mat G_x (3x3)
static const double sobel_g_x_mat[] = {
    //  [1 0 -1]
    1,
    0,
    -1,
    //  [2 0 -2]
    2,
    0,
    -2,
    //  [1 0 -1]
    1,
    0,
    -1,
};

// Mat G_y (3x3)
static const double sobel_g_y_mat[] = {
    //  [1 2 1]
    1,
    2,
    1,
    //  [0 0 0]
    0,
    0,
    0,
    //  [-1 -2 -1]
    -1,
    -2,
    -1,
};

static inline uint8_t map_result(double in)
{
    // Cap
    if (in > 255)
        return 255;
    if (in < 0)
        return 0;
    return (uint8_t)in;
}

// sobel_seq performs a sequential sobel operator.
int sobel_seq(ppm_image *in_img, ppm_image *out_img);

// sobel_omp performs a sequential sobel operator.
// int sobel_omp(ppm_image *in_img, ppm_image *out_img);

#endif