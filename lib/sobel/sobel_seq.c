#include <stdint.h>

#include "lib/img/types.h"
#include "sobel.h"
#include "lib/img/operations.h"

// Sequential sobel implementation
void sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset, int last_call)
{
    if(offset != in_img->size_y - 3){

        uint8_t *pixel_line = read_pixel_line(in_img);
        if(pixel_line == NULL)
        {
            fprintf(stderr, "'%s' couldn't be loaded", in_img->filename);
            return ;
        }
        memcpy(in_img->data + (offset +4) * in_img->size_x, pixel_line, in_img->size_x);
        free(pixel_line);
    }
    sobel_on_line(in_img->data, out_img->data, offset, in_img->size_x, in_img->size_y, out_img->size_x, out_img->size_y);

}
