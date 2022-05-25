#ifndef __IMG_TYPES_H__
#define __IMG_TYPES_H__
#include <stdint.h>
#include <stdio.h>

// ppm_image stores an image as a matrix in form of X rows and Y columns.
typedef struct
{
    // Image metadata
    uintmax_t size_x;
    uintmax_t size_y;

    // pixel depth
    uint8_t depth; // 1 -> Grayscale; 2 -> Color

    // file to save / read from
    FILE *fp;
    char *filename;
    
    // Data
    uint8_t *data; // we only load greyscale images, but pixels are RGB; [y * size_x + x + color_offset]; color_offset is 0,1,2
} ppm_image;

#endif