#include <stdint.h>
#include <omp.h>

#include "lib/img/types.h"
#include "sobel.h"


// Sequential sobel implementation
int sobel(ppm_image *in_img, ppm_image *out_img)
{

// FIXME add input validation
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < omp_get_num_threads(); i++)
    {
        const int num_threads = omp_get_num_threads();
        const uintmax_t chunk_size = ((in_img->size_x - 2) / num_threads);
        const uintmax_t chunk_loss = (in_img->size_x - 2) % num_threads;
        // Have a deterministic for loop chunking our workload.

        uintmax_t x = chunk_size * (uintmax_t)i + (i == 0 ? 0 : chunk_loss);
        uintmax_t x_to = chunk_size * ((uintmax_t)i + 1) + chunk_loss;
        uintmax_t y_to = in_img->size_y - 2;
        // Actual computation
        for (; x < x_to; x++)
        {
            // Not neccessary to further accelerate it as the workload is already evenly distributed with the
            for (uintmax_t y = 0; y < y_to; y++)
            {
                out_img->data[y * out_img->size_x + x] = sobel_combined_op(in_img, x, y);
            }
        }
    }

    return 0;
}
