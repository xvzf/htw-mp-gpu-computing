#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "operations.h"

// load_image loads an image in PPM format.
ppm_image *load_image(const char *filename)
{
    ppm_image *img;
    FILE *fp;
    char buf[3];

    //open PPM file for reading
    fp = fopen(filename, "rb");
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
    if (buf[0] != 'P' || buf[1] != '6')
    {
        fprintf(stderr, "[!] Header mismatch in file '%s', expected P6 (Bitmap)", filename);
        return NULL;
    }

    img = (ppm_image *)malloc(sizeof(ppm_image));
    img->depth = 3;
    if (!img)
    {
        fprintf(stderr, "[!] Failed to alloc memory");
        return NULL;
    }

    //read image size information
    if (fscanf(fp, "%llu %llu", &img->size_x, &img->size_y) != 2)
    {
        fprintf(stderr, "[!] couldn't extract image size for '%s'\n", filename);
        return NULL;
    }

    //check if image is grayscale (only option supported)
    int max_pix_value;
    if (fscanf(fp, "%d", &max_pix_value) != 1)
    {
        fprintf(stderr, "[!] Couldn't extract max_pix_value for %s", filename);
        return NULL;
    }
    if (max_pix_value > 255)
    {
        fprintf(stderr, "[!] invalid color range in '%s' (expected 8bit)", filename);
        return NULL;
    }

    // Mem alloc for data
    img->data = (uint8_t *)malloc(sizeof(uint8_t) * 3 * img->size_x * img->size_y);
    if (!img->data)
    {
        fprintf(stderr, "[!] Failed to alloc memory");
        return NULL;
    }

    //read pixel data from file, assume we're having a grayscale image -> only read one pixel value
    if (fread(img->data, 3 * img->size_x, img->size_y, fp) != img->size_y)
    {
        fprintf(stderr, "'%s' couldn't be loaded", filename);
        return NULL;
    }

    fclose(fp);
    return img;
}

// save_image stores an image in the PPM P6 format
// FIXME error handling!
int save_image(const char *filename, ppm_image *img)
{
    FILE *fp;

    if (img->depth != 3)
    {
        return -1;
    }

    //open PPM file for reading
    fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "[!] Unable to open file '%s'\n", filename);
        return -1;
    }

    // write header
    fprintf(fp, "P6\n");
    //write size
    fprintf(fp, "%llu %llu\n", img->size_x, img->size_y);
    // color depth
    fprintf(fp, "255\n");
    // Write data
    fwrite(img->data, 3 * img->size_x, img->size_y, fp);

    fclose(fp);
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
    out->size_x = in->size_x;
    out->size_y = in->size_y;
    out->depth = 1;

    for (uint64_t x = 0; x < in->size_x; x++)
    {
        for (uint64_t y = 0; y < in->size_y; y++)
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
    out->size_x = in->size_x;
    out->size_y = in->size_y;
    out->depth = 3;

    for (uint64_t x = 0; x < in->size_x; x++)
    {
        for (uint64_t y = 0; y < in->size_y; y++)
        {
            for (uint8_t d = 0; d < out->depth; d++)
            {
                out->data[y * out->size_x * out->depth + x + d] = in->data[y * in->size_x * in->depth + x];
            }
        }
    }

    return out;
}