HEADERS = img.h mat_helper.h
OBJECTS = sobel.o img.o mat_helper.o

CC := gcc-11

default: sobel

%.o: %.c $(HEADERS)
	${CC} -c $< -o $@ -fsanitize=leak

sobel: $(OBJECTS)
	${CC} $(OBJECTS) -o $@ -fopenmp -fsanitize=leak

clean:
	-rm -f $(OBJECTS)
	-rm -f sobel

covert_src_to_ppm:
	convert ${SOURCE} -colorspace Gray  ${TARGET}.ppm