#ifndef __IMG_OPERATIONS_H__
#define __IMG_OPERATIONS_H__

#include <stdint.h>
#include "types.h"


// load_image loads an image in PPM format
ppm_image *load_image(const char *filename, int depth);

// save_image stores an image in the PPM P6 format
int save_image(ppm_image *img);

// takes a color pixel line of size depth * size_x and converts it to grayscale pixel line
static uint8_t* color_to_gray_line(uint8_t *pixel_line, uint8_t depth, uintmax_t size_x)
{
    static uint8_t *pixel_gray = NULL;
    if(pixel_gray == NULL) {
        pixel_gray = (uint8_t *) malloc(size_x);
    }

    if (depth != 3)
    {
        free(pixel_gray);
        return NULL;
    }

    for (uintmax_t x = 0; x < size_x; x++)
        pixel_gray[x] = pixel_line[x * 3];

    return pixel_gray;
}

// reads a line of color pixels from image, converts them to grayscale and returns grayscale line with size size_x
static uint8_t* read_pixel_line(ppm_image *img)
{
    static uint8_t *pixel_line = NULL;
    if(pixel_line == NULL){
        pixel_line = (uint8_t*) malloc(img->depth * img->size_x);
    }
    uint8_t *gray_line;

    if (fread(pixel_line, img->depth * img->size_x, 1, img->fp) != 1)
    {
        fprintf(stderr, "'%s' couldn't be loaded", img->filename);
        free(pixel_line);
        return NULL;
    }

    gray_line = color_to_gray_line(pixel_line, img->depth, img->size_x);
    return gray_line;
}

// new_image creates a new (empty) image with the given parameters
ppm_image *new_image(const char *filename, uintmax_t size_x, uintmax_t size_y, uint8_t depth);

#endif