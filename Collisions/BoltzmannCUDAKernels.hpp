#ifndef BOLTZMANN_CUDA_KERNELS_HPP
#define BOLTZMANN_CUDA_KERNELS_HPP

#include <cmath>
#include <cuComplex.h> 

using Complex = cuDoubleComplex; // Note: real(z) = z.x, imag(z) = z.y

__host__ __device__ inline Complex cuCconj(const Complex &z);

// We need to define two versions of cuCexp: one for the host and one for the device
// The device version uses CUDA intrinsics, while the host version uses the standard library
__host__ inline Complex cuCexp(const Complex &z);
__device__ inline Complex cuCexp(const Complex &z);

__host__ __device__ inline cuDoubleComplex cuCscale(const cuDoubleComplex &z, const double scale);

// Declarations for the CUDA kernels
__global__ void copy_to_complex(Complex * result, const double * input, const int N);

__global__ void compute_alpha_times_f_hat(Complex * alpha1_times_f_hat, Complex * alpha2_times_f_hat,
                                        const Complex * alpha1, const Complex * f_hat, 
                                        const int N_gl, const int N_spherical, 
                                        const int Nvx, const int Nvy, const int Nvz, 
                                        const double scale);

__global__ void hadamard_product(Complex * result, const Complex *x, const Complex *y, const int N);

__global__ void compute_beta2_times_f_hat(Complex * beta2_times_f_hat, const double * beta2, 
                                        const Complex * f_hat, const int N, const double scale);

__global__ void compute_Q_total(double * Q, const Complex * Q_gain, 
                                const Complex * beta2_times_f_hat, const Complex * f, 
                                const int N);

#endif // BOLTZMANN_CUDA_KERNELS_HPP




