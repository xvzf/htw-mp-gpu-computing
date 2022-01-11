#ifndef __IMG_OPERATIONS_H__
#define __IMG_OPERATIONS_H__

#include <stdio.h>
#include <stdint.h>

// ppm_image stores an image as a matrix in form of X rows and Y columns.
typedef struct
{
    // Image metadata
    uint64_t size_x;
    uint64_t size_y;

    // pixel depth
    uint8_t depth; // 1 -> Grayscale; 2 -> Color

    // Data
    uint8_t *data; // we only load greyscale images, but pixels are RGB; [y * size_x + x + color_offset]; color_offset is 0,1,2
} ppm_image;

// load_image loads an image in PPM format
ppm_image *load_image(const char *filename);

// save_image stores an image in the PPM P6 format
int save_image(const char *filename, ppm_image *img);

// color_to_gray reduces the RGB color range to a single dimension.
ppm_image *color_to_gray(ppm_image *in);

// greay_to_color explodes a single dimension to RGB
ppm_image *gray_to_color(ppm_image *in);

#endif