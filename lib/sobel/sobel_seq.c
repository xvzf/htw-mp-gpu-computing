#include <stdint.h>

#include "lib/img/types.h"
#include "sobel.h"

// Sequential sobel implementation
int sobel(ppm_image *in_img, ppm_image *out_img)
{
    // FIXME add input validation

    int max_x = in_img->size_x - 2;
    int max_y = in_img->size_y - 2;
    for (uintmax_t x = 0; x < max_x; x++)
    {
        for (uintmax_t y = 0; y < max_y; y++) {

            out_img->data[y * out_img->size_x + x] = sobel_combined_op(in_img, x, y);
        }
    }


    return 0;
}