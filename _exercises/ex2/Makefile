all: \
	ex2_1a.out \
	ex2_1b.out \
	ex2_2.out \
	ex2_2_pat.out \
	ex2_3.out

%.out: %.cu
	nvcc $*.cu -o $*.out

pdf:
	pandoc Readme.md --pdf-engine=xelatex -o Readme.pdf
