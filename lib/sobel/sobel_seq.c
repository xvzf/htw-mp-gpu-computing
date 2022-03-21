#include <stdint.h>

#include "lib/img/types.h"
#include "sobel.h"

// Sequential sobel implementation
int sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset)
{
    // FIXME add input validation
    sobel_on_line(in_img->data, out_img->data, offset, in_img->size_x, in_img->size_y, out_img->size_x, out_img->size_y);

    /*int max_x = in_img->size_x - 2;
    for (uintmax_t x = 0; x < max_x; x++)
    {
        out_img->data[offset * out_img->size_x + x] = sobel_combined_op(in_img, x, offset);
    }
*/
    return 0;
}