#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "lib/img/types.h"
#include "lib/img/operations.h"
#include "lib/sobel/sobel.h"
uint8_t constant_maps_host[MAP_SIZE] = {0};

// main entrypoint
int main(int argc, char **argv)
{

    double first_start_time = omp_get_wtime();
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

    if(in_img == NULL)
    {
        fprintf(stderr, "[!] In-Image could not be loaded!");
        return -1;
    }
    printf("Image size: %lu %lu (%lu pixels) data pointer: %p depth: %u\n", in_img->size_x, in_img->size_y, in_img->size_x * in_img->size_y, in_img->data, in_img->depth);
    ppm_image *out_img = new_image(out_path, in_img->size_x - 2, in_img->size_y - 2, 1);

    if(out_img == NULL)
    {
        fprintf(stderr, "[!] Out-Image could not be created!");
        free(in_img->data);
        fclose(in_img->fp);
        free(in_img->filename);
        free(in_img);
        return -1;
    }
    init_host_maps();
    //print_host_maps();

    uint8_t *pixel_line;
    // load first three lines 
    for(intmax_t i = 0; i < 3; i++)
    {
        pixel_line = read_pixel_line(in_img);
        if(pixel_line == NULL)
        {
            fprintf(stderr, "'%s' couldn't be loaded", in_img->filename);
            return -1;
        }
        memcpy(in_img->data + i * in_img->size_x, pixel_line, in_img->size_x);
        free(pixel_line);
    }

    uint8_t **device_out_ptr = malloc(sizeof(uint8_t*));
    *device_out_ptr = NULL;
    // Perform sobel operator on first line
    double start_time = omp_get_wtime();
    uint8_t *_device_out = sobel(in_img, out_img, 0, device_out_ptr);
   // ret =
    if (ret != EXIT_SUCCESS)
    {
        fprintf(stderr, "[!] Sobel operator failed to run!");
    }

    if(_device_out != NULL){
        printf("herpiderpi\n");
        write_to_out_img(out_img, 0, _device_out);
    }
    for(intmax_t i = 0; i < in_img->size_y - 3; i++)
    {

        //pixel_line = read_pixel_line(in_img);
        //if(pixel_line == NULL)
        //{
        //    fprintf(stderr, "'%s' couldn't be loaded", in_img->filename);
        //    return -1;
        //}
        //memcpy(in_img->data + (i+3) * in_img->size_x, pixel_line, in_img->size_x);
        //free(pixel_line);
        // Perform sobel operator
        //ret = sobel(in_img, out_img, i + 1, device_out_ptr);
        _device_out = sobel(in_img, out_img, i+ 1, device_out_ptr);
        if (ret != EXIT_SUCCESS)
        {
            fprintf(stderr, "[!] Sobel operator failed to run!");
        }
      //  printf("main: %p %p %p\n", _device_out, device_out, *device_out_ptr);
      //  if(device_out != NULL){
      //      printf("derp\n");
      //      write_to_out_img(out_img, i + 1, device_out);
      //  }else{
      //      printf("device_out was null\n");
      //      if(_device_out != NULL){
      //          printf("derp\n");
      //          write_to_out_img(out_img, i + 1, _device_out);
      //      }else{
      //          printf("_device_out was null\n");
      //          if(*device_out_ptr != NULL){
      //              printf("derp\n");
      //              write_to_out_img(out_img, i + 1, *device_out_ptr);
      //          }else{
      //              printf("*device_out_ptr was null\n");
      //          }
      //      }
      //  }
    }

    printf("Compute took %lfms\n", (omp_get_wtime() - start_time) * 1000);
    // Save image
    ret += save_image(out_img);
    printf("Saved img\n");
    // Free up resources
    free(in_img->data);
    free(out_img->data);
    fclose(out_img->fp);
    fclose(in_img->fp);
    free(in_img->filename);
    free(out_img->filename);
    free(in_img);
    free(out_img);
    printf("Complete took %lfms\n", (omp_get_wtime() - first_start_time) * 1000);
    return ret;
}
