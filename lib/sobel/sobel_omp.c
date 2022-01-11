#include <math.h>
#include <omp.h>

#include "lib/img/operations.h"
#include "sobel.h"

// Sequential sobel implementation
int sobel_omp(ppm_image *in_img, ppm_image *out_img)
{
    // FIXME add input validation
    #pragma omp parralel for
    for (uintmax_t x = 0; x < in_img->size_x - 2; x++)
    {
        // Not neccessary to further accelerate it as the workload is already evenly distributed with the
        for (uintmax_t y = 0; y < in_img->size_y - 2; y++)
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

            double g = sqrt(pow(g_x, 2.0f) + pow(g_y, 2.0f));

            out_img->data[y * out_img->size_x + x] = map_result(g);
        }
    }

    return 0;
}