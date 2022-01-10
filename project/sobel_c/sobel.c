#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "img.h"

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
    }
    in_path = argv[1];
    out_path = argv[2];

    // Load image
    in_img = load_image(in_path);

    return 0;
}