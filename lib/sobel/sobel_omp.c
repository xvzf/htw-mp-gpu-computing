#include <stdint.h>
#include <omp.h>

#include "lib/img/types.h"
#include "lib/img/operations.h"
#include "sobel.h"


// Parallel sobel implementation
void sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset, int last_call)
{

    if(offset != in_img->size_y - 3){

        uint8_t *pixel_line = read_pixel_line(in_img);
        if(pixel_line == NULL)
        {
            fprintf(stderr, "'%s' couldn't be loaded", in_img->filename);
            return ;
        }
        memcpy(in_img->data + (offset + 4) * in_img->size_x, pixel_line, in_img->size_x);
        free(pixel_line);
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < omp_get_num_threads(); i++)
    {
        const int num_threads = omp_get_num_threads();
        const uintmax_t chunk_size = ((in_img->size_x - 2) / num_threads);
        const uintmax_t chunk_loss = (in_img->size_x - 2) % num_threads;
        // Have a deterministic for loop chunking our workload.

        uintmax_t x = chunk_size * (uintmax_t)i + (i == 0 ? 0 : chunk_loss);
        uintmax_t x_to = chunk_size * ((uintmax_t)i + 1) + chunk_loss;
        // Actual computation
        for (; x < x_to; x++)
        {
                out_img->data[offset * out_img->size_x + x] = sobel_combined_op(in_img, x, offset);        }
    }
}