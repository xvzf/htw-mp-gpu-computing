#ifndef __SOBEL_H__
#define __SOBEL_H__

#include <stdint.h>
#include <math.h>
#include "lib/img/types.h"

#define MAP_SIZE (256*256)
#ifdef __CUDACC__
__constant__ uint8_t constant_maps[MAP_SIZE];

#endif

static uint8_t *device_out;
extern uint8_t constant_maps_host[MAP_SIZE];

static void init_host_maps(){
    for(int i = 0; i < MAP_SIZE;++i){
        constant_maps_host[i] = sqrt(i);
    }
}
static void print_host_maps(){
    uint8_t last_value = constant_maps_host[0];
    for(int i = 0; i < MAP_SIZE;++i){
        if(last_value != constant_maps_host[i]) {
            printf("%u ", constant_maps_host[i]);
            last_value = constant_maps_host[i];
        }
    }
}
#ifdef __cplusplus
extern "C"
{
#endif
    // sobel runs the sobel operator.
    uint8_t *sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset, uint8_t **return_device_out);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif
// sobel runs the sobel operator.
int write_to_out_img(ppm_image *out_img, intmax_t offset, uint8_t *_device_out);
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
typedef float calc_t;

#ifdef __CUDACC__
__device__
#endif
    static inline uint8_t
    map_result(calc_t in)
{
    // Cap
    if (in > 255)
        return 255;
    if (in < 0)
        return 0;
    // Cap
    return (uint8_t)(in);
}


#ifdef __CUDACC__
__device__
#endif
static inline uint8_t
sobel_combined_op_data(uint8_t *data, uintmax_t x, uintmax_t y, uintmax_t size_x, uintmax_t size_y)
{

    uintmax_t tmp_top_row = y * size_x;
    uintmax_t tmp_middle_row = (y + 1) * size_x;
    uintmax_t tmp_bottom_row = (y + 2) * size_x;

    calc_t top_left = data[tmp_top_row + x];
    //calc_t top = data[tmp_top_row + (x + 1)];
    calc_t top_right = data[tmp_top_row + (x + 2)];
    //calc_t left = data[tmp_middle_row + x];
    //calc_t right = data[tmp_middle_row + (x + 2)];
    calc_t bottom_left = data[tmp_bottom_row + x];
    //calc_t bottom = data[tmp_bottom_row + (x + 1)];
    calc_t bottom_right = data[tmp_bottom_row + (x + 2)];


    // Apply sobel operator
    calc_t g_x = top_left + 2 * data[tmp_middle_row + x] + bottom_left - (top_right + 2 * data[tmp_middle_row + (x + 2)] + bottom_right);

    calc_t g_y = top_left + 2 * data[tmp_top_row + (x + 1)] + top_right - (bottom_left + 2 * data[tmp_bottom_row + (x + 1)] + bottom_right);
    calc_t tmp = g_x * g_x + g_y * g_y;
    if(tmp < 1) return 0;
    if(tmp >= MAP_SIZE) return 255;

#ifdef __CUDACC__
    return constant_maps[(int)tmp];
#else
    return constant_maps_host[(int)tmp];
#endif
}

#ifdef __CUDACC__
__device__
#endif
static inline uint8_t
sobel_combined_op_data_three_rows(uint8_t *data_first_row, uint8_t *data_second_row, uint8_t *data_third_row, uintmax_t x, uintmax_t y, uintmax_t size_x, uintmax_t size_y)
{

    /*uintmax_t tmp_top_row = y * size_x;
    uintmax_t tmp_middle_row = (y + 1) * size_x;
    uintmax_t tmp_bottom_row = (y + 2) * size_x;*/

    calc_t top_left = data_first_row[x];
    //calc_t top = data[tmp_top_row + (x + 1)];
    calc_t top_right = data_first_row[ (x + 2)];
    //calc_t left = data[tmp_middle_row + x];
    //calc_t right = data[tmp_middle_row + (x + 2)];
    calc_t bottom_left = data_third_row[x];
    //calc_t bottom = data[tmp_bottom_row + (x + 1)];
    calc_t bottom_right = data_third_row[(x + 2)];


    // Apply sobel operator
    calc_t g_x = top_left + 2 * data_second_row[x] + bottom_left - (top_right + 2 * data_second_row[(x + 2)] + bottom_right);

    calc_t g_y = top_left + 2 * data_first_row[(x + 1)] + top_right - (bottom_left + 2 * data_third_row[(x + 1)] + bottom_right);
    calc_t tmp = g_x * g_x + g_y * g_y;
    if(tmp < 1) return 0;
    if(tmp >= MAP_SIZE) return 255;

#ifdef __CUDACC__
    return constant_maps[(int)tmp];
#else
    return constant_maps_host[(int)tmp];
#endif
}

#ifdef __CUDACC__
__device__
#endif
static inline uint8_t
sobel_on_line(uint8_t *data, uint8_t  *out_data, uintmax_t y, uintmax_t size_x, uintmax_t size_y, uintmax_t out_size_x, uintmax_t out_size_y)
{
    int max_x = size_x - 2;
    calc_t old_top_right;
    calc_t old_bottom_right;
    calc_t old_top;
    calc_t old_right;
    calc_t old_bottom;
    calc_t older_right;
    {
        uintmax_t x = 0;
        uintmax_t tmp_top_row = y * size_x;
        uintmax_t tmp_middle_row = (y + 1) * size_x;
        uintmax_t tmp_bottom_row = (y + 2) * size_x;

        calc_t top_left = data[tmp_top_row + x];
        calc_t top_right = data[tmp_top_row + (x + 2)];
        calc_t bottom_left = data[tmp_bottom_row + x];
        calc_t bottom_right = data[tmp_bottom_row + (x + 2)];
        calc_t top = data[tmp_top_row + (x + 1)];
        calc_t left = data[tmp_middle_row + x];
        calc_t right = data[tmp_middle_row + (x + 2)];
        calc_t bottom = data[tmp_bottom_row + (x + 1)];


        // Apply sobel operator
        calc_t g_x = top_left + 2 * left + bottom_left -
                     (top_right + 2 * right + bottom_right);

        calc_t g_y = top_left + 2 * top + top_right -
                     (bottom_left + 2 * bottom + bottom_right);

        // Map double -> 0..255
        calc_t tmp = g_x * g_x + g_y * g_y;
        if(tmp < 1) {
            out_data[y * out_size_x + x] = 0;
        }
        else if(tmp >= MAP_SIZE) {
            out_data[y * out_size_x + x] = 255;
        }
        else {
#ifdef __CUDACC__
            out_data[y * out_size_x + x] = constant_maps[(int)tmp];
#else
            out_data[y * out_size_x + x] = constant_maps_host[(int)tmp];
#endif
        }
        old_right = right;
    }
    {
        uintmax_t x = 1;
        uintmax_t tmp_top_row = y * size_x;
        uintmax_t tmp_middle_row = (y + 1) * size_x;
        uintmax_t tmp_bottom_row = (y + 2) * size_x;

        calc_t top_left = data[tmp_top_row + x];
        calc_t top_right = data[tmp_top_row + (x + 2)];
        calc_t bottom_left = data[tmp_bottom_row + x];
        calc_t bottom_right = data[tmp_bottom_row + (x + 2)];
        calc_t top = data[tmp_top_row + (x + 1)];
        calc_t left = data[tmp_middle_row + x];
        calc_t right = data[tmp_middle_row + (x + 2)];
        calc_t bottom = data[tmp_bottom_row + (x + 1)];


        // Apply sobel operator
        calc_t g_x = top_left + 2 * left + bottom_left -
                     (top_right + 2 * right + bottom_right);

        calc_t g_y = top_left + 2 * top + top_right -
                     (bottom_left + 2 * bottom + bottom_right);

        // Map double -> 0..255
        calc_t tmp = g_x * g_x + g_y * g_y;
        if(tmp < 1) {
            out_data[y * out_size_x + x] = 0;
        }
        else if(tmp >= MAP_SIZE) {
            out_data[y * out_size_x + x] = 255;
        }
        else {
#ifdef __CUDACC__
            out_data[y * out_size_x + x] = constant_maps[(int)tmp];
#else
            //out_data[y * out_size_x + x] = map_result(sqrt(g_x * g_x + g_y *g_y));
            out_data[y * out_size_x + x] = constant_maps_host[(int) tmp];
#endif
        }
        older_right = old_right;
        old_top_right = top_right;
        old_bottom_right = bottom_right;
        old_top = top;
        old_right = right;
        old_bottom = bottom;
    }
    for (uintmax_t x = 2; x < max_x; x++)
    {

        uintmax_t tmp_top_row = y * size_x;
        uintmax_t tmp_middle_row = (y + 1) * size_x;
        uintmax_t tmp_bottom_row = (y + 2) * size_x;

        calc_t top_left = old_top;
        calc_t top_right = data[tmp_top_row + (x + 2)];
        calc_t bottom_left = old_bottom;
        calc_t bottom_right = data[tmp_bottom_row + (x + 2)];
        calc_t top = old_top_right;
        calc_t left = older_right;
        calc_t right = data[tmp_middle_row + (x + 2)];
        calc_t bottom = old_bottom_right;


        // Apply sobel operator
        calc_t g_x = top_left + 2 * left + bottom_left -
                     (top_right + 2 * right + bottom_right);

        calc_t g_y = top_left + 2 * top + top_right -
                     (bottom_left + 2 * bottom + bottom_right);

        // Map double -> 0..255
        calc_t tmp = g_x * g_x + g_y * g_y;
        if(tmp < 1) {
            out_data[y * out_size_x + x] = 0;
        }
        else if(tmp >= MAP_SIZE) {
            out_data[y * out_size_x + x] = 255;
        }
        else {
#ifdef __CUDACC__
            out_data[y * out_size_x + x] = constant_maps[(int)tmp];
#else
            //out_data[y * out_size_x + x] = map_result(sqrt(g_x * g_x + g_y *g_y));
            out_data[y * out_size_x + x] = constant_maps_host[(int) tmp];
#endif
        }
        older_right = old_right;
        old_top_right = top_right;
        old_bottom_right = bottom_right;
        old_top = top;
        old_right = right;
        old_bottom = bottom;
    }
    return 0;
}


// sobel_compined_op wraps the sobel_combined_op_data for direct ppm_image input
#ifndef __CUDACC__
static inline uint8_t sobel_combined_op(ppm_image *in_img, uintmax_t x, uintmax_t y)
{
    return sobel_combined_op_data(in_img->data, x, y, in_img->size_x, in_img->size_y);
}
#endif

#endif