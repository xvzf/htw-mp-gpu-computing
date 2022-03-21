#include <stdint.h>

#include "lib/img/types.h"
#include "sobel.h"
#include "lib/img/operations.h"

// Sequential sobel implementation
uint8_t *sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset, uint8_t **return_device_out)
{
    if(offset != in_img->size_y - 3){

        uint8_t *pixel_line = read_pixel_line(in_img);
        if(pixel_line == NULL)
        {
            fprintf(stderr, "'%s' couldn't be loaded", in_img->filename);
            return NULL;
        }
        memcpy(in_img->data + (offset +4) * in_img->size_x, pixel_line, in_img->size_x);
        free(pixel_line);
    }
    // FIXME add input validation
    sobel_on_line(in_img->data, out_img->data, offset, in_img->size_x, in_img->size_y, out_img->size_x, out_img->size_y);

    /*int max_x = in_img->size_x - 2;
    for (uintmax_t x = 0; x < max_x; x++)
    {
        out_img->data[offset * out_img->size_x + x] = sobel_combined_op(in_img, x, offset);
    }
*/
    return NULL;
}


int write_to_out_img(ppm_image *out_img, intmax_t offset, uint8_t *_device_out){
    // is done in sobel func
}