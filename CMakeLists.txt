cmake_minimum_required(VERSION 3.18)

# Name for the project
# Note that for CUDA codes, it is better to specify
# CUDA as a language than use "find_package(CUDA REQUIRED)"
# as the latter is deprecated.
project(Boltzmann_Benchmark LANGUAGES C CXX)

find_package(OpenMP REQUIRED)
#find_package(CUDAToolkit REQUIRED)

add_library(Boltzmann)
add_subdirectory(Utilities)
add_subdirectory(Quadratures)
add_subdirectory(Collisions)

# This has to be set before project()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

#----------------------------------------------------------------------------#
# Options
#----------------------------------------------------------------------------#
option(USE_AVX512 "Enable AVX-512 architecture flags" OFF)
option(USE_SANTITIZER "Enable leak detection" OFF)

# Set the minimum standard for the languages
set(CMAKE_C_STANDARD    11 CACHE STRING "C language standard")
set(CMAKE_CXX_STANDARD  11 CACHE STRING "C++ language standard")
set(CMAKE_CUDA_STANDARD 11 CACHE STRING "CUDA language standard")

option(CMAKE_C_STANDARD_REQUIRED    "Require the C language standard to set"    ON)
option(CMAKE_CXX_STANDARD_REQUIRED  "Require the C++ language standard to set"  ON)
option(CMAKE_CUDA_STANDARD_REQUIRED "Require the CUDA language standard to set" ON)

option(CMAKE_C_EXTENSIONS    "Enable/disable extensions, e.g. -std=gnu11 vs. -std=c11"     OFF)
option(CMAKE_CXX_EXTENSIONS  "Enable/disable extensions, e.g. -std=gnu++11 vs. -std=c++11" OFF)
option(CMAKE_CUDA_EXTENSIONS "Enable/disable extensions" OFF)

# Create the executable targets for the different codes
add_executable(maxwell_bkw_fftw_ex maxwell_bkw_fftw.cpp)
add_executable(fftw_benchmark_ex fftw_benchmark.cpp)
add_executable(loop_benchmark_cpu_ex loop_benchmarks_cpu.cpp)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# Create interface target with compiler flags
add_library(compile-options INTERFACE)

#----------------------------------------------------------------------------------------#
# macro that checks if flag if supported for C, if so add to compile-options
#----------------------------------------------------------------------------------------#
macro(ADD_C_FLAG_IF_AVAIL FLAG)
    if(NOT "${FLAG}" STREQUAL "")
        # create a variable for checking the flag if supported, e.g.:
        #   -fp-model=precise --> c_fp_model_precise
        string(REGEX REPLACE "^-" "c_" FLAG_NAME "${FLAG}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")

        check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
        if(${FLAG_NAME})
            target_compile_options(compile-options INTERFACE
                $<$<COMPILE_LANGUAGE:C>:${FLAG}>)
        endif()
    endif()
endmacro()

#----------------------------------------------------------------------------------------#
# macro that checks if flag if supported for C++, if so add to foo-compile-options
#----------------------------------------------------------------------------------------#
macro(ADD_CXX_FLAG_IF_AVAIL FLAG)
    if(NOT "${FLAG}" STREQUAL "")
        # create a variable for checking the flag if supported, e.g.:
        #   -fp-model=precise --> cxx_fp_model_precise
        string(REGEX REPLACE "^-" "cxx_" FLAG_NAME "${FLAG}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")

        # runs check to see flag is supported by compiler
        check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
        if(${FLAG_NAME})
            target_compile_options(compile-options INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
        endif()
    endif()
endmacro()

#----------------------------------------------------------------------------------------#
# macro that checks if flag if supported for C and C++
#----------------------------------------------------------------------------------------#
macro(ADD_FLAGS_IF_AVAIL)
    foreach(FLAG ${ARGN})
        add_c_flag_if_avail("${FLAG}")
        add_cxx_flag_if_avail("${FLAG}")
    endforeach()
endmacro()


# standard flags for C and C++
add_flags_if_avail("-O3")

# "new" keyword doesn't exist in C so no need to check
add_cxx_flag_if_avail("-faligned-new")

# OpenMP SIMD-only (supported by GCC)
add_flags_if_avail("-fopenmp-simd")

# enable runtime leak detection
if(USE_SANITIZER)
    add_flags_if_avail("-fsanitize=leak")

    # emit warnings that this feature is not available
    if(NOT c_fsanitize_leak)
        message(WARNING "Sanitizer is not available for selected C compiler")
    endif()

    if(NOT cxx_fsanitize_leak)
        message(WARNING "Sanitizer is not available for selected C++ compiler")
    endif()
endif()

# check for AVX-512 flags
if(USE_AVX512)
    if(CMAKE_C_COMPILER_ID MATCHES "Intel")
        add_flags_if_avail("-xMIC-AVX512")
    else()
        # these flags are supported by newer GCC versions
        add_flags_if_avail("-mavx512f" "-mavx512pf" "-mavx512er" "-mavx512cd")
    endif()
endif()

# Specify the location of the header files for the argument parser
target_include_directories(maxwell_bkw_fftw_ex PRIVATE Include)
target_include_directories(fftw_benchmark_ex PRIVATE Include)

# Link the libraries to the code
# Cray compilers automatically provide the link commands to fftw3 and other math libraries
target_link_libraries(maxwell_bkw_fftw_ex PRIVATE compile-options Boltzmann OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
target_link_libraries(fftw_benchmark_ex PRIVATE compile-options Boltzmann OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
