#ifndef __IMG_OPERATIONS_H__
#define __IMG_OPERATIONS_H__

#include <stdint.h>
#include "types.h"


// load_image loads an image in PPM format
ppm_image *load_image(const char *filename, int depth);

// save_image stores an image in the PPM P6 format
int save_image(ppm_image *img);

// reads a line of color pixels from image, converts them to grayscale and returns grayscale line with size size_x
uint8_t* read_pixel_line(ppm_image *img);

// takes a color pixel line of size depth * size_x and converts it to grayscale pixel line
uint8_t* color_to_gray_line(uint8_t *pixel_line, uint8_t depth, uintmax_t size_x);

// new_image creates a new (empty) image with the given parameters
ppm_image *new_image(const char *filename, uintmax_t size_x, uintmax_t size_y, uint8_t depth);

#endif