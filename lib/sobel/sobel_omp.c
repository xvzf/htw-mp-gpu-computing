#include <stdint.h>
#include <omp.h>

#include "lib/img/types.h"
#include "sobel.h"

// Sequential sobel implementation
int sobel_omp(ppm_image *in_img, ppm_image *out_img)
{
// FIXME add input validation
#pragma omp parallel for
    for (uintmax_t x = 0; x < in_img->size_x - 2; x++)
    {
        // Not neccessary to further accelerate it as the workload is already evenly distributed with the
        for (uintmax_t y = 0; y < in_img->size_y - 2; y++)
        {
            out_img->data[y * out_img->size_x + x] = sobel_combined_op(in_img, x, y);
        }
    }

    return 0;
}