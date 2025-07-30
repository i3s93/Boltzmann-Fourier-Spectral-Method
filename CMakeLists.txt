cmake_minimum_required(VERSION 3.18)

# This has to be set before project()
# Otherwise the wrong version of __CUDA__ARCH__ is set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Name for the project
# Note that for CUDA codes, it is better to specify
# CUDA as a language than use "find_package(CUDA REQUIRED)"
# as the latter is deprecated.
project(Boltzmann_Benchmark LANGUAGES C CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create the executable targets for the different codes
add_executable(maxwell_bkw_fftw_ex maxwell_bkw_fftw.cpp)
add_executable(maxwell_bkw_cuda_ex maxwell_bkw_cuda.cu)

add_executable(fftw_benchmark_ex fftw_benchmark.cpp)
add_executable(cufft_benchmark_ex cufft_benchmark.cu)
add_executable(loop_benchmark_cpu_ex loop_benchmark_cpu.cpp)
add_executable(loop_benchmark_gpu_ex loop_benchmark_gpu.cpp)

# Get FFTW paths from environment
if(DEFINED ENV{FFTW_ROOT})
    set(FFTW_ROOT $ENV{FFTW_ROOT})
    set(FFTW_INCLUDE_DIR "${FFTW_ROOT}/include")
    set(FFTW_LIBRARY_DIR "${FFTW_ROOT}/lib")
    message(STATUS "FFTW include path: ${FFTW_INCLUDE_DIR}")
    message(STATUS "FFTW library path: ${FFTW_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "FFTW_ROOT environment variable is not set.")
endif()

# Look for FFTW using the path pointed to by the user
find_library(FFTW3_LIBRARY fftw3 HINTS ${FFTW_LIBRARY_DIR})
find_library(FFTW3_THREADS_LIBRARY fftw3_omp HINTS ${FFTW_LIBRARY_DIR})

if(NOT FFTW3_LIBRARY)
    message(FATAL_ERROR "Could not find fftw3 in ${FFTW_LIBRARY_DIR}")
endif()

if(NOT FFTW3_THREADS_LIBRARY)
    message(WARNING "Could not find fftw3_threads in ${FFTW_LIBRARY_DIR}")
endif()

## Don't mess with setting environment variables... just pass them as options
## during the invocation of cmake... (same with cuTensorNet)
##
## Get the path to the cutensor library from the environment
#if((DEFINED ENV{CUTENSOR_INCLUDE_DIR}) AND (DEFINED ENV{CUTENSOR_LIBRARY_DIR}))
#    set(CUTENSOR_INCLUDE_DIR "${CUTENSOR_INCLUDE_DIR}")
#    set(CUTENSOR_LIBRARY_DIR "${CUTENSOR_LIBRARY_DIR}")
#    message(STATUS "cutensor include path: ${CUTENSOR_INCLUDE_DIR}")
#    message(STATUS "cutensor library path: ${CUTENSOR_LIBRARY_DIR}")
#else()
#    message(FATAL_ERROR "cuTensor environment variables are not set.")
#endif()

# Get the path to cuquantum from the environment
if(DEFINED ENV{CUQUANTUM_ROOT})
    set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})
    set(CUQUANTUM_INCLUDE_DIR "${CUQUANTUM_ROOT}/include")
    set(CUQUANTUM_LIBRARY_DIR "${CUQUANTUM_ROOT}/lib")
    message(STATUS "cuquantum include path: ${CUQUANTUM_INCLUDE_DIR}")
    message(STATUS "cuquantum library path: ${CUQUANTUM_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "CUQUANTUM_ROOT environment variable is not set.")
endif()

# First create the Boltzmann library target
add_library(Boltzmann)

# Add include directory
target_include_directories(Boltzmann PUBLIC Include)
target_include_directories(Boltzmann PUBLIC ${FFTW_INCLUDE_DIR})
target_include_directories(Boltzmann PUBLIC ${CUTENSOR_INCLUDE_DIR})
target_include_directories(Boltzmann PUBLIC ${CUQUANTUM_INCLUDE_DIR})

# Add the directories where the cuquantum and cutensor libraries are stored
target_link_directories(Boltzmann PUBLIC ${CUTENSOR_LIBRARY_DIR})
target_link_directories(Boltzmann PUBLIC ${CUQUANTUM_LIBRARY_DIR})

# Link libraries (FFTW, OpenMP, CUDA, etc.)
target_link_libraries(Boltzmann PUBLIC ${FFTW3_LIBRARY} ${FFTW3_THREADS_LIBRARY})
target_link_libraries(Boltzmann PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
target_link_libraries(Boltzmann PUBLIC CUDA::cufft)
target_link_libraries(Boltzmann PUBLIC cutensor cutensornet)

# Optionally add threads support if available
if(FFTW3_THREADS_LIBRARY)
    target_link_libraries(Boltzmann PUBLIC ${FFTW3_THREADS_LIBRARY})
endif()

add_subdirectory(Utilities)
add_subdirectory(Quadratures)
add_subdirectory(Collisions)

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

if(CMAKE_C_COMPILER_ID MATCHES "NVHPC" AND CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
    message(STATUS "Using NVIDIA HPC compilers (C and C++)")

    add_flags_if_avail("-O3" "-fast" "-mp" "-gpu=cc80,sm_80")

else()
    # standard flags for C and C++
    add_flags_if_avail("-O3" "-funroll-loops" "-ftree-vectorize" "-ffast-math")

    # "new" keyword doesn't exist in C so no need to check
    add_cxx_flag_if_avail("-faligned-new")

    # OpenMP SIMD-only (supported by GCC)
    add_flags_if_avail("-fopenmp-simd")

endif()

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

# Pass the compile options to the Boltzmann library
target_link_libraries(Boltzmann PUBLIC compile-options)

target_link_libraries(maxwell_bkw_fftw_ex PRIVATE Boltzmann)
target_link_libraries(maxwell_bkw_cuda_ex PRIVATE Boltzmann)

target_link_libraries(fftw_benchmark_ex PRIVATE Boltzmann)
target_link_libraries(cufft_benchmark_ex PRIVATE Boltzmann)
target_link_libraries(loop_benchmark_cpu_ex PRIVATE Boltzmann)
target_link_libraries(loop_benchmark_gpu_ex PRIVATE Boltzmann)








