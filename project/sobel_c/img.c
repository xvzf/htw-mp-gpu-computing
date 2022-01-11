#include "img.h"

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
    if (!img)
    {
        fprintf(stderr, "[!] Failed to alloc memory");
        return NULL;
    }

    //read image size information
    if (fscanf(fp, "%ld %ld", &img->size_x, &img->size_y) != 2)
    {
        fprintf(stderr, "[!] couldn't extract image size for '%s'\n", filename);
        return NULL;
    }

    //check if image is greyscale (only option supported)
    int max_pix_value;
    if (fscanf(fp, "%d", &max_pix_value) != 1)
    {
        fprintf(stderr, "[!] Couldn't extract max_pix_value for %s", filename);
        return NULL;
    }
    if (max_pix_value > 255)
    {
        fprintf(stderr, "[!] invalid color range (expected 8bit)", filename);
        return NULL;
    }

    // Mem alloc for data
    img->data = (uint8_t *)malloc(sizeof(uint8_t) * 3 * img->size_x * img->size_y);
    if (!img->data)
    {
        fprintf(stderr, "[!] Failed to alloc memory");
        return NULL;
    }

    //read pixel data from file, assume we're having a greyscale image -> only read one pixel value
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
    fprintf(fp, "%ld %ld\n", img->size_x, img->size_y);
    // color depth
    fprintf(fp, "255\n");
    // Write data
    fwrite(img->data, 3 * img->size_x, img->size_y, fp);

    fclose(fp);
    return 0;
}