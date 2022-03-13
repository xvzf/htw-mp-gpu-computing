#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "operations.h"

ppm_image *new_image(const char *filename, uintmax_t size_x, uintmax_t size_y, uint8_t depth)
{
    ppm_image *out;

    if ((depth != 1) && (depth != 3))
    {
        return NULL;
    }

    // Allocate
    out = (ppm_image *)malloc(sizeof(ppm_image));

    out->filename = malloc(sizeof(char) * (strlen(filename) + 1));
    memcpy(out->filename, filename, sizeof(char) * (strlen(filename) + 1));
    out->fp = fopen(filename, "wb+");
    if (!out->fp)
    {
        fprintf(stderr, "[!] Unable to open file '%s'\n", filename);
        return NULL;
    }
    out->data = (uint8_t *)malloc(size_x * size_y * depth);
    out->size_x = size_x;
    out->size_y = size_y;
    out->depth = depth;

    return out;
}

// load_image loads an image in PPM format.
ppm_image *load_image(const char *filename, int depth)
{
    ppm_image *img;
    FILE *fp;
    char buf[3];

    //open PPM file for reading
    fp = fopen(filename, "rb+");
    if (!fp)
    {
        fprintf(stderr, "[!] Unable to open file '%s'\n", filename);
        return NULL;
    }

    // Read header
    if (!fgets(buf, sizeof(buf), fp))
    {
        perror(filename);
        return NULL;
    }
    // check header
    if (buf[0] != 'P' || (buf[1] != '6' && buf[1] != '5'))
    {
        fprintf(stderr, "[!] Header mismatch in file '%s', expected P6 (Bitmap)", filename);
        return NULL;
    }

    img = (ppm_image *)malloc(sizeof(ppm_image));
    img->depth = buf[1] == '5' ? 1: 3; // P5 -> Grayscale P6 -> RGB
    if (!img)
    {
        fprintf(stderr, "[!] Failed to alloc memory");
        return NULL;
    }

    img->filename = malloc(sizeof(char) * (strlen(filename) + 1));
    memcpy(img->filename, filename, sizeof(char) * (strlen(filename) + 1));
    img->fp = fp;
    if (!img->fp)
    {
        fprintf(stderr, "[!] Unable to open file '%s'\n", img->filename);
        return NULL;
    }

    //read image size information
    if (fscanf(img->fp, "%ju %ju", &img->size_x, &img->size_y) != 2)
    {
        fprintf(stderr, "[!] couldn't extract image size for '%s'\n", img->filename);
        return NULL;
    }

    //check if image is grayscale (only option supported)
    int max_pix_value;
    if (fscanf(img->fp, "%d", &max_pix_value) != 1)
    {
        fprintf(stderr, "[!] Couldn't extract max_pix_value for %s", img->filename);
        return NULL;
    }
    if (max_pix_value > 255)
    {
        fprintf(stderr, "[!] invalid color range in '%s' (expected 8bit)", img->filename);
        return NULL;
    }

    // Mem alloc for data
    img->data = (uint8_t *)malloc(sizeof(uint8_t) * img->depth * img->size_x * img->size_y);
    if (!img->data)
    {
        fprintf(stderr, "[!] Failed to alloc memory");
        return NULL;
    }

    uint8_t *pixel_line;
    for(intmax_t i = 0; i < img->size_y; i++)
    {
        pixel_line = read_pixel_line(img);
        if(pixel_line == NULL)
        {
            fprintf(stderr, "'%s' couldn't be loaded", img->filename);
            return NULL;
        }
        memcpy(img->data + i * img->size_x, pixel_line, img->size_x);
        free(pixel_line);
    }

    //read pixel data from file, assume we're having a grayscale image -> only read one pixel value
/*    if (fread(img->data, img->depth * img->size_x, img->size_y, img->fp) != img->size_y)
    {
        fprintf(stderr, "'%s' couldn't be loaded", img->filename);
        return NULL;
    }*/

    fclose(fp);

    // Convert to grayscale if not already present
/*    if (depth == 1 && img->depth == 3) {
        ppm_image *img_grey = color_to_gray(img);
        free(img->data);
        free(img);
        return img_grey;
    }*/

    return img;
}

uint8_t* read_pixel_line(ppm_image *img)
{
    uint8_t *pixel_line = malloc(img->depth * img->size_x);
    uint8_t *gray_line;

    if (fread(img->data, img->depth * img->size_x, 1, img->fp) != 1)
    {
        fprintf(stderr, "'%s' couldn't be loaded", img->filename);
        free(pixel_line);
        return NULL;
    }

    gray_line = color_to_gray_line(pixel_line, img->depth, img->size_x);
    free(pixel_line);

    return gray_line;
}

uint8_t* color_to_gray_line(uint8_t *pixel_line, uint8_t depth, uint8_t size_x)
{
    uint8_t *pixel_gray;

    pixel_gray = malloc(size_x);

    if (depth != 3)
    {
        free(pixel_gray);
        return NULL;
    }

    for (uintmax_t x = 0; x < size_x; x++)
        pixel_gray[x] = pixel_line[depth + x * 3];

    return pixel_gray;
}

// save_image stores an image in the PPM P6 format
// FIXME error handling!
int save_image(ppm_image *img)
{

    // write header
    if (img->depth == 3)
        fprintf(img->fp, "P6\n");
    else
        fprintf(img->fp, "P5\n");

    //write size
    fprintf(img->fp, "%ju %ju\n", img->size_x, img->size_y);
    // color depth
    fprintf(img->fp, "255\n");
    // Write data
    fwrite(img->data, img->depth * img->size_x, img->size_y, img->fp);

    fclose(img->fp);
    img->fp = fopen(img->filename, "rb+");
    return 0;
}

// color_to_gray reduces the RGB color range to a single dimension.
ppm_image *color_to_gray(ppm_image *in)
{
    ppm_image *out;

    if (in->depth != 3)
    {
        return NULL;
    }

    // Allocate
    out = (ppm_image *)malloc(sizeof(ppm_image));
    out->data = (uint8_t *)malloc(in->size_x * in->size_y);

    out->filename = malloc(sizeof(char) * (strlen(in->filename) + 1));
    memcpy(out->filename, in->filename, sizeof(char) * (strlen(in->filename) + 1));
    out->fp = fopen(out->filename, "rb+");
    if (!out->fp)
    {
        fprintf(stderr, "[!] Unable to open file '%s'\n", out->filename);
        return NULL;
    }

    out->size_x = in->size_x;
    out->size_y = in->size_y;
    out->depth = 1;

    for (uintmax_t x = 0; x < in->size_x; x++)
    {
        for (uintmax_t y = 0; y < in->size_y; y++)
        {
            out->data[y * out->size_x + x] = in->data[y * in->size_x * in->depth + x * 3];
        }
    }
    return out;
}

// greay_to_color explodes a single dimension to RGB
ppm_image *gray_to_color(ppm_image *in)
{
    ppm_image *out;

    if (in->depth != 1)
    {
        return NULL;
    }

    // Allocate
    out = (ppm_image *)malloc(sizeof(ppm_image));
    out->data = (uint8_t *)malloc(3 * in->size_x * in->size_y);

    out->filename = malloc(sizeof(char) * (strlen(in->filename) + 1));
    memcpy(out->filename, in->filename, sizeof(char) * (strlen(in->filename) + 1));
    out->fp = fopen(out->filename, "rb+");
    if (!out->fp)
    {
        fprintf(stderr, "[!] Unable to open file '%s'\n", out->filename);
        return NULL;
    }

    out->size_x = in->size_x;
    out->size_y = in->size_y;
    out->depth = 3;

    for (uintmax_t x = 0; x < in->size_x; x++)
    {
        for (uintmax_t y = 0; y < in->size_y; y++)
        {
            for (uint8_t d = 0; d < out->depth; d++)
            {
                out->data[y * out->size_x * out->depth + x + d] = in->data[y * in->size_x * in->depth + x];
            }
        }
    }

    return out;
}