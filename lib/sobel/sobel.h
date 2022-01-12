#ifndef __SOBEL_H__
#define __SOBEL_H__

#include <stdint.h>
#include <math.h>
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
int sobel_omp(ppm_image *in_img, ppm_image *out_img);

// sobel_combined_op is an internal function used in more than one implementation
static inline uint8_t sobel_combined_op(ppm_image *in_img, uintmax_t x, uintmax_t y)
{
    // Apply sobel operator
    double g_x =
        // First row
        (double)in_img->data[y * in_img->size_x + x] * sobel_g_x_mat[0 + 0] + (double)in_img->data[(y + 1) * in_img->size_x + x + 1] * sobel_g_x_mat[0 + 1] + (double)in_img->data[(y + 2) * in_img->size_x + x + 2] * sobel_g_x_mat[0 + 2]
        // Second row
        + (double)in_img->data[y * in_img->size_x + x] * sobel_g_x_mat[3 + 0] + (double)in_img->data[(y + 1) * in_img->size_x + x + 1] * sobel_g_x_mat[3 + 1] + (double)in_img->data[(y + 2) * in_img->size_x + x + 2] * sobel_g_x_mat[3 + 2]
        // Third row
        + (double)in_img->data[y * in_img->size_x + x] * sobel_g_x_mat[6 + 0] + (double)in_img->data[(y + 1) * in_img->size_x + x + 1] * sobel_g_x_mat[6 + 1] + (double)in_img->data[(y + 2) * in_img->size_x + x + 2] * sobel_g_x_mat[6 + 2];

    double g_y =
        // First row
        (double)in_img->data[y * in_img->size_x + x] * sobel_g_y_mat[0 + 0] + (double)in_img->data[(y + 1) * in_img->size_x + x + 1] * sobel_g_y_mat[0 + 1] + (double)in_img->data[(y + 2) * in_img->size_x + x + 2] * sobel_g_y_mat[0 + 2]
        // Second row
        + (double)in_img->data[y * in_img->size_x + x] * sobel_g_y_mat[3 + 0] + (double)in_img->data[(y + 1) * in_img->size_x + x + 1] * sobel_g_y_mat[3 + 1] + (double)in_img->data[(y + 2) * in_img->size_x + x + 2] * sobel_g_y_mat[3 + 2]
        // Third row
        + (double)in_img->data[y * in_img->size_x + x] * sobel_g_y_mat[6 + 0] + (double)in_img->data[(y + 1) * in_img->size_x + x + 1] * sobel_g_y_mat[6 + 1] + (double)in_img->data[(y + 2) * in_img->size_x + x + 2] * sobel_g_y_mat[6 + 2];

    // Map double -> 0..255
    return map_result(sqrt(pow(g_x, 2.0f) + pow(g_y, 2.0f)));
}

#endif