cmake_minimum_required(VERSION 3.9)
project(MDN)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED COMPONENTS system filesystem iostreams)
INCLUDE_DIRECTORIES(${INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${LINK_DIRECTORIES} ${Boost_LIBRARY_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(./gnuplot-iostream/)
include_directories(./fann/src/)
include_directories(./fann/src/include/)

add_executable(example_nn
        example_nn.cpp
        fann/src/floatfann.c
        fann/src/include/fann_cpp.h)
target_link_libraries(example_nn ${Boost_LIBRARIES})

add_executable(example_mdn
        MixtureDensityNetwork.h
        MixtureDensityNetwork.cpp
        GaussianMixtureModel.h
        GaussianMixtureModel.cpp
        DirTools.h
        DirTools.cpp
        example_mdn.cpp)
target_link_libraries(example_mdn ${Boost_LIBRARIES})