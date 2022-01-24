#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include "lib/img/types.h"
#include "lib/img/operations.h"
#include "lib/sobel/sobel.h"


// main entrypoint
int main(int argc, char **argv)
{
    int ret = 0;

    // Input output
    char *in_path, *out_path;

    // Check in args
    if (argc != 3)
    {
        fprintf(stderr, "[!] invalid args\n\n\t usage: %s <in> <out>\n", argv[0]);
        return -1;
    }
    in_path = argv[1];
    out_path = argv[2];

    // Load image
    ppm_image *in_img = load_image(in_path, 1);
    ppm_image *out_img = new_image(in_img->size_x - 2, in_img->size_y - 2, 1);

    // Perform sobel operator
    ret = sobel(in_img, out_img);
    if(ret) {
        // Save image
        ret += save_image(out_path, out_img);
    } else {
        fprintf(stderr, "[!] Sobel operator failed to run!");
    }

    // Free up resources
    free(in_img->data);
    free(out_img->data);
    free(in_img);
    free(out_img);
    return ret;
}
