#ifndef __SOBEL_H__
#define __SOBEL_H__

#include <stdint.h>
#include <math.h>
#include "lib/img/types.h"

#ifdef __cplusplus
extern "C"
{
#endif
    // sobel runs the sobel operator.
    int sobel(ppm_image *in_img, ppm_image *out_img);
#ifdef __cplusplus
}
#endif

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

#ifdef __CUDACC__
__device__
#endif
    static inline uint8_t
    map_result(double in)
{
    // Cap
    if (in > 255)
        return 255;
    if (in < 0)
        return 0;
    return (uint8_t)in;
}

// sobel_combined_op is an internal function used in more than one implementation
#ifdef __CUDACC__
__device__
#endif
    static inline uint8_t
    sobel_combined_op_data(uint8_t *data, uintmax_t x, uintmax_t y, uintmax_t size_x, uintmax_t size_y)
{

    uintmax_t tmp_top_row = y * size_x;
    uintmax_t tmp_middle_row = (y + 1) * size_x;
    uintmax_t tmp_bottom_row = (y + 2) * size_x;

    double top_left = data[tmp_top_row + x];
    //double top = data[tmp_top_row + (x + 1)];
    double top_right = data[tmp_top_row + (x + 2)];
    //double left = data[tmp_middle_row + x];
    //double right = data[tmp_middle_row + (x + 2)];
    double bottom_left = data[tmp_bottom_row + x];
    //double bottom = data[tmp_bottom_row + (x + 1)];
    double bottom_right = data[tmp_bottom_row + (x + 2)];


    // Apply sobel operator
    double g_x = top_left + 2 * data[tmp_middle_row + x] + bottom_left - (top_right + 2 * data[tmp_middle_row + (x + 2)] + bottom_right);

    double g_y = top_left + 2 * data[tmp_top_row + (x + 1)] + top_right - (bottom_left + 2 * data[tmp_bottom_row + (x + 1)] + bottom_right);

    // Map double -> 0..255
    return map_result(sqrt(g_x * g_x + g_y * g_y));
}

// sobel_compined_op wraps the sobel_combined_op_data for direct ppm_image input
#ifndef __CUDACC__
static inline uint8_t sobel_combined_op(ppm_image *in_img, uintmax_t x, uintmax_t y)
{
    return sobel_combined_op_data(in_img->data, x, y, in_img->size_x, in_img->size_y);
}
#endif

#endif