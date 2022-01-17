#ifndef __IMG_OPERATIONS_H__
#define __IMG_OPERATIONS_H__

#include <stdint.h>
#include "types.h"


// load_image loads an image in PPM format
ppm_image *load_image(const char *filename, int depth);

// save_image stores an image in the PPM P6 format
int save_image(const char *filename, ppm_image *img);

// color_to_gray reduces the RGB color range to a single dimension.
ppm_image *color_to_gray(ppm_image *in);

// greay_to_color explodes a single dimension to RGB
ppm_image *gray_to_color(ppm_image *in);

// new_image creates a new (empty) image with the given parameters
ppm_image *new_image(uintmax_t size_x, uintmax_t size_y, uint8_t depth);

#endif