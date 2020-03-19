all: lud_omp lud_omp_offload

one:
	cd omp; make one
out:
	cd omp; make out

run:
	./omp/one.bin -s 8000
run_out:
	./omp/out.bin -s 8000


lud_omp:
	cd omp; make

lud_omp_offload:
	cd omp; make -f Makefile.offload

clean:
	cd omp; make clean ; make -f Makefile.offload clean
