cmake_minimum_required(VERSION 2.8)
project(liblinear)

add_subdirectory(blas)
add_subdirectory(matlab)

add_library(
        linearStatic STATIC
        linear.cpp
        linear.h
        tron.cpp
        tron.h
)

add_executable(train train.c)

target_link_libraries(
        train
        linearStatic
        blas
)

add_executable(
        predict
        predict.c
)
target_link_libraries(
        predict
        linearStatic
        blas
)
