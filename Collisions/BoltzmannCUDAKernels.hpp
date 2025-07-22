#ifndef BOLTZMANN_CUDA_KERNELS_HPP
#define BOLTZMANN_CUDA_KERNELS_HPP

#include <cmath>
#include <limits>
#include <cuComplex.h>

template<typename T>
__host__ __device__ T eps() { return T(0); }

template<>
__host__ __device__ float eps<float>() { return 1.1920929e-7f; }

template<>
__host__ __device__ double eps<double>() { return 2.220446049250313e-16; }

template<typename T>
__host__ __device__
T sincc(T x) {
    T epsilon = eps<T>();
    T xp = x + epsilon;
#if defined(__CUDA_ARCH__)
    // Device code: use CUDA device math
    return sin(xp) / xp;
#else
    // Host code: use std::sin
    return std::sin(xp) / xp;
#endif
}

__host__ __device__ inline Complex cuCconj(const cuDoubleComplex &z) {
    return make_cuDoubleComplex(z.x, -z.y);
}

__host__ __device__ inline cuDoubleComplex cuCscale(const cuDoubleComplex &z, const double scale) {
    return make_cuDoubleComplex(z.x * scale, z.y * scale);
}

// Declarations for the CUDA kernels
__global__ void copy_to_complex(
    cuDoubleComplex * __restrict__ result, 
    const double * __restrict__ input, 
    const int N);

__global__ void compute_alpha_times_f_hat(
    cuDoubleComplex * __restrict__ alpha1_times_f_hat,
    cuDoubleComplex * __restrict__ alpha2_times_f_hat,
    const cuDoubleComplex * __restrict__ f_hat,
    const double * __restrict__ gl_nodes,
    const int * __restrict__ lx, const int * __restrict__ ly, const int * __restrict__ lz,
    const double * __restrict__ sx, const double * __restrict__ sy, const double * __restrict__ sz,
    const int N_gl, const int N_spherical, const int Nvx, const int Nvy, const int Nvz,
    const double pi, const double L, const double fft_scale);

__global__ void hadamard_product(
    cuDoubleComplex * __restrict__ result, 
    const cuDoubleComplex *  __restrict__ x, 
    const cuDoubleComplex *  __restrict__ y, 
    const int N);

__global__ void atomic_tensor_contraction(
    cuDoubleComplex * __restrict__ Q_gain_hat,
    const double * __restrict__ gl_nodes,
    const double * __restrict__ gl_wts,
    const double * __restrict__ spherical_wts,
    const int * __restrict__ lx,
    const int * __restrict__ ly,
    const int * __restrict__ lz,
    const cuDoubleComplex * __restrict__ transform_prod_hat,
    const int N_gl, const int N_spherical, 
    const int Nvx, const int Nvy, const int Nvz,
    const double gamma, const double b_gamma, 
    const double pi, const double L, const double fft_scale);

__global__ void compute_beta2_times_f_hat(
    cuDoubleComplex * __restrict__ beta2_times_f_hat, 
    const cuDoubleComplex * __restrict__ f_hat, 
    const double * __restrict__ gl_wts,
    const double * __restrict__ gl_nodes,
    const int * __restrict__ lx,
    const int * __restrict__ ly,
    const int * __restrict__ lz,
    const int N_gl, const int Nvx, const int Nvy, const int Nvz,
    const double gamma, const double b_gamma,
    const double pi, const double L, 
    const double fft_scale);

__global__ void compute_Q_total(
    double * __restrict__ Q, 
    const cuDoubleComplex * __restrict__ Q_gain, 
    const cuDoubleComplex * __restrict__ beta2_times_f_hat, 
    const cuDoubleComplex * __restrict__ f, 
    const int N);

#endif // BOLTZMANN_CUDA_KERNELS_HPP




