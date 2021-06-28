add_library(
        matlab STATIC
        libsvmread.c
        libsvmwrite.c
        linear_model_matlab.c
        linear_model_matlab.h
        predict.c
        train.c
)