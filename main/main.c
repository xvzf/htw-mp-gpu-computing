#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include "lib/img/operations.h"
#include "lib/matrix/helper.h"

// main entrypoint
int main(int argc, char **argv)
{
    // Input output
    char *in_path, *out_path;
    ppm_image *in_img;

    // Check in args
    if (argc != 3)
    {
        fprintf(stderr, "[!] invalid args\n\n\t usage: %s <in> <out>\n", argv[0]);
        return -1;
    }
    in_path = argv[1];
    out_path = argv[2];

    // Load image
    in_img = load_image(in_path);

    ppm_image* test = color_to_gray(in_img);

    save_image(out_path, gray_to_color(test));

    return 0;
}