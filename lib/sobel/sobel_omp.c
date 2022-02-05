#include <stdint.h>
#include <omp.h>

#include "lib/img/types.h"
#include "sobel.h"

#define OMP_SPLIT_CHUNKS 32 // Not ideal as there's no auto-detection of CPU cores but required for OMP

// Sequential sobel implementation
int sobel(ppm_image *in_img, ppm_image *out_img)
{
    const uintmax_t chunk_size = ((in_img->size_x - 2) / OMP_SPLIT_CHUNKS);
    const uintmax_t chunk_loss = (in_img->size_x - 2) % OMP_SPLIT_CHUNKS;

// FIXME add input validation
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < OMP_SPLIT_CHUNKS; i++)
    {
        // Have a deterministic for loop chunking our workload.

        uintmax_t x = chunk_size * (uintmax_t)i + (i == 0 ? 0 : chunk_loss);
        uintmax_t x_to = chunk_size * ((uintmax_t)i + 1) + chunk_loss;

        // Actual computation
        for (; x < x_to; x++)
        {
            // Not neccessary to further accelerate it as the workload is already evenly distributed with the
            for (uintmax_t y = 0; y < in_img->size_y - 2; y++)
            {
                out_img->data[y * out_img->size_x + x] = sobel_combined_op(in_img, x, y);
            }
        }
    }

    return 0;
}
