all: \
	ex3_1.out \
	ex3_3.out \
	ex3_1_pat.out \
	ex3_3_pat.out \
	ex3_2.cout \

all: ex3_1.out

CC := gcc-11
CXX := g++-11

%.out: %.cu
	nvcc $*.cu -o $*.out -lomp

%.cout: %.c
	${CC} $*.c -o $*.cout -fopenmp

%.cppout: %.c
	${CXX} $*.c -o $*.cout -fopenmp

clean:
	rm *.cout
	rm *.out

pdf:
	pandoc Readme.md --pdf-engine=xelatex -o Readme.pdf