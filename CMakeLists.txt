project("liblinear")

cmake_minimum_required(VERSION 2.8)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(PkgConfig REQUIRED)

SET( LINKING_MODE SHARED)
#SET( LINKING_MODE STATIC)

MESSAGE("Linking mode: ${LINKING_MODE}")

#add_definitions(
#	-std=c++11
#)

SET( CMAKE_CXX_FLAGS_RELEASE " -fPIC -O3 -ffast-math -funroll-all-loops -fpeel-loops -ftracer -funswitch-loops -funit-at-a-time" CACHE STRING "" FORCE)
SET( CMAKE_CXX_FLAGS_DEBUG "${GOP_WARNING_FLAGS} -fPIC -g3 -O0" CACHE STRING "" FORCE )
SET( CMAKE_C_FLAGS_RELEASE " -fPIC -O3 -ffast-math -funroll-all-loops -fpeel-loops -ftracer -funswitch-loops -funit-at-a-time" CACHE STRING "" FORCE)
SET( CMAKE_C_FLAGS_DEBUG "${GOP_WARNING_FLAGS} -fPIC -g3 -O0" CACHE STRING "" FORCE )

find_package(OpenMP)
if (OPENMP_FOUND)
	MESSAGE("USING OPENMP")
	add_definitions(-DENABLED_OPENMP)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(Boost COMPONENTS regex REQUIRED)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
FIND_PACKAGE(Boost COMPONENTS filesystem REQUIRED)
#FIND_PACKAGE(Boost COMPONENTS gil REQUIRED)
#FIND_PACKAGE(Boost COMPONENTS thread REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system REQUIRED)
#FIND_PACKAGE(Boost COMPONENTS iostreams REQUIRED)

include_directories(
        "${Boost_INCLUDE_DIRS}"
)

link_directories(
        "/opt/local/lib"
        "/usr/local/lib"
)

# defining linear library name and include path
SET(LINEAR_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/linear)
SET(LINEAR_LIBRARIES linear)
INCLUDE_DIRECTORIES(${LINEAR_INCLUDE_PATH})

# defining linear library name and include path
SET(BLAS_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/blas)
SET(BLAS_LIBRARIES blas)
INCLUDE_DIRECTORIES(${BLAS_INCLUDE_PATH})

# add lib subdirecory with libraries
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# add apps subdirecories with applications
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/apps/train)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/apps/predict)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/apps/predict_stream)

