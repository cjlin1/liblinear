ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS = train predict 
lib_LTLIBRARIES = liblinear.la

liblinear_la_SOURCES = tron.cpp linear.cpp blas/blasp.h 	blas/daxpy.c 	blas/ddot.c 	blas/dnrm2.c blas/dscal.c
liblinear_la_LDFLAGS = -static -avoid-version

include_HEADERS = linear.h

train_SOURCES = train.cpp
train_LDADD   = liblinear.la -lstdc++ 

predict_SOUCES = predict.cpp
predict_LDADD  = liblinear.la -lstdc++
