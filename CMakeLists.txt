cmake_minimum_required(VERSION 3.15)
project(app VERSION 1.0.0 LANGUAGES CXX CUDA)

add_executable(app "")

message("-- Check for OpenCV")
find_package(OpenCV REQUIRED)
if (OPENCV_FOUND)
    message("-- Found OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS}")
    message("    Libraries:  ${OpenCV_LIBS}")
else (OPENCV_FOUND)
    message(FATAL_ERROR "Can't find OpenCV. Please set OPENCV_INCLUDE_DIR & OPENCV_LIBRARY")
endif (OPENCV_FOUND)

message("-- Check for OpenMP")
find_package(OpenMP)
if (OPENMP_FOUND)
    message("-- Found OpenMP ${OpenMP_VERSION}: ${OpenMP_INCLUDE_DIRS}")
else (OPENMP_FOUND)
    message(FATAL_ERROR "Can't find OpenMP.")
endif (OPENMP_FOUND)

target_sources(app
        PUBLIC
        src/main.cpp
        src/middle.h
        src/middle.cpp
        src/saxpy.cuh
        src/saxpy.cu)

target_include_directories(app
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/>
        PRIVATE
        ${OPENCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        )

target_link_libraries(app
        PUBLIC
        ${OpenCV_LIBS})

target_compile_options(app
        PUBLIC
        # CXX FLAGS
        $<$<BOOL:OPENMP_FOUND>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wall>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wextra>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-pedantic>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-function>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-switch>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-parameter>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-std=c++17>
        # CXX RELEASE ONLY FLAGS
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Ofast>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-ffast-math>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-march=native>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-pedantic>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-result>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-unused-parameter>>
        $<$<CONFIG:Release>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-Wno-return-type>>
        # CUDA FLAGS
        $<$<BOOL:OPENMP_FOUND>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-std=c++14>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--compiler-options>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--default-stream per-thread>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--expt-extended-lambda>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcudart>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcudrand>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcudadevrt>
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-lcublas_device>
        #CUDA DEBUG ONLY FLAGS
        $<$<CONFIG:Debug>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-O0>>
        $<$<CONFIG:Debug>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-arch=sm_30 -gencode=arch=compute_50,code=sm_50>>
        $<$<CONFIG:Debug>:$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:--ptxas-options=-v>>
        )

set_target_properties(app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(app PROPERTIES POSITION_INDEPENDENT_CODE ON)