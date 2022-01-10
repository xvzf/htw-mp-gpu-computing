#ifndef __SOBEL_IMG_H__
#define __SOBEL_IMG_H__

#include <stdio.h>
#include <stdlib.h>

// ppm_image stores an image as a matrix in form of X rows and Y columns.
typedef struct
{
    // Image metadata
    uint64_t size_x;
    uint64_t size_y;

    // Data
    uint8_t *data; // we only load greyscale images, but pixels are RGB; [y * size_x + x + color_offset]; color_offset is 0,1,2
} ppm_image;

// load_image loads an image in PPM format
ppm_image *load_image(const char *filename);

#endif