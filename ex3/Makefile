all: \
	ex3_1.out \
	ex3_2.cout

all: ex3_1.out

GCC := gcc-11

%.out: %.cu
	nvcc $*.cu -o $*.out -lomp

%.cout: %.c
	${GCC} $*.c -o $*.cout -fopenmp

clean:
	rm *.cout
	rm *.out

pdf:
	pandoc Readme.md --pdf-engine=xelatex -o Readme.pdf
