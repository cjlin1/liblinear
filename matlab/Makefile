# This Makefile is used under Linux

MATLABDIR ?= /usr/local/matlab
CXX ?= g++
#CXX = g++-3.3
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -I$(MATLABDIR)/extern/include -I..

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC="$(CXX)" CXX="$(CXX)" CFLAGS="$(CFLAGS)" CXXFLAGS="$(CFLAGS)"
# comment the following line if you use MATLAB on a 32-bit computer
MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

all:	matlab

matlab:	binary

octave:
	@echo "please type make under Octave"
binary: train.$(MEX_EXT) predict.$(MEX_EXT) libsvmread.$(MEX_EXT) libsvmwrite.$(MEX_EXT)

train.$(MEX_EXT): train.c ../linear.h ../tron.cpp ../linear.cpp linear_model_matlab.c \
		../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
	$(MEX) $(MEX_OPTION) train.c ../tron.cpp ../linear.cpp linear_model_matlab.c \
		../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c

predict.$(MEX_EXT): predict.c ../linear.h ../tron.cpp ../linear.cpp linear_model_matlab.c \
		../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
	$(MEX) $(MEX_OPTION) predict.c ../tron.cpp ../linear.cpp linear_model_matlab.c \
		../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c

libsvmread.$(MEX_EXT):	libsvmread.c
	$(MEX) $(MEX_OPTION) libsvmread.c

libsvmwrite.$(MEX_EXT):	libsvmwrite.c
	$(MEX) $(MEX_OPTION) libsvmwrite.c

clean:
	rm -f *~ *.o *.mex* *.obj
