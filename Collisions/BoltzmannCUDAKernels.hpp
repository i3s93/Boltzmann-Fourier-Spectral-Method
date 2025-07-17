#ifndef BOLTZMANN_CUDA_KERNELS_HPP
#define BOLTZMANN_CUDA_KERNELS_HPP

#include <cmath>
#include <cuComplex.h> 

using Complex = cuDoubleComplex; // Note: real(z) = z.x, imag(z) = z.y

__host__ __device__ inline Complex cuCconj(const Complex &z) {
    return make_cuDoubleComplex(z.x, -z.y);
}

// We need to define two versions of cuCexp: one for the host and one for the device
// The device version uses CUDA intrinsics, while the host version uses the standard library
__host__ __device__ inline Complex cuCexp(const Complex &z) {
    double real = z.x;
    double imag = z.y;
#if defined(__CUDA_ARCH__)
    double exp_real = exp(real);
    return make_cuDoubleComplex(exp_real * cos(imag), exp_real * sin(imag));
#else
    double exp_real = std::exp(real);
    return make_cuDoubleComplex(exp_real * std::cos(imag), std::sin(imag));
#endif
}

__host__ __device__ inline cuDoubleComplex cuCscale(const cuDoubleComplex &z, const double scale) {
    return make_cuDoubleComplex(z.x * scale, z.y * scale);
}


// Declarations for the CUDA kernels
__global__ void copy_to_complex(Complex * result, const double * input, const int N);

__global__ void compute_alpha_times_f_hat(Complex * alpha1_times_f_hat, Complex * alpha2_times_f_hat,
                                        const Complex * alpha1, const Complex * f_hat, 
                                        const int N_gl, const int N_spherical, 
                                        const int Nvx, const int Nvy, const int Nvz, 
                                        const double scale);

__global__ void hadamard_product(Complex * result, const Complex *x, const Complex *y, const int N);

__global__ void atomic_tensor_contraction(Complex * Q_gain_hat, 
                                        const Complex * radial_term, const Complex * spherical_wts, 
                                        const Complex * beta1, const Complex * transform_prod_hat, 
                                        const int N_gl, const int N_spherical, 
                                        const int Nvx, const int Nvy, const int Nvz, 
                                        const double scale);

__global__ void compute_beta2_times_f_hat(Complex * beta2_times_f_hat, const double * beta2, 
                                        const Complex * f_hat, const int N, const double scale);

__global__ void compute_Q_total(double * Q, const Complex * Q_gain, 
                                const Complex * beta2_times_f_hat, const Complex * f, 
                                const int N);

#endif // BOLTZMANN_CUDA_KERNELS_HPP




