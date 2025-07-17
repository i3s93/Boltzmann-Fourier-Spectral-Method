#include "BoltzmannCUDAKernels.hpp"


// CUDA kernel to copy a double array to a Complex array
__global__ void copy_to_complex(Complex * result, const double * input, const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += num_threads) {
        result[i].x = input[i];
        result[i].y = 0.0;
    }
}

/*

// Kernel that computes alpha1_times_f_hat and alpha2_times_f_hat
// We map (r,s) to thread blocks and the loops over (i,j,k) are handled by the threads within each block
// Note: When launching this kernel, the grid size should be set to N_gl * N_spherical
__global__ void compute_alpha_times_f_hat(Complex * alpha1_times_f_hat, Complex * alpha2_times_f_hat,
                                        const Complex * alpha1, const Complex * f_hat, 
                                        const int N_gl, const int N_spherical, 
                                        const int Nvx, const int Nvy, const int Nvz, 
                                        const double scale){
    // Compute the block index and backout the (r,s) indices
    int block_idx = blockIdx.x;
    int r = block_idx / N_spherical;
    int s = block_idx % N_spherical;
    const int grid_size = Nvx * Nvy * Nvz;
    const int base = (r * N_spherical + s) * grid_size;

    for (int idx = threadIdx.x; idx < grid_size; idx += blockDim.x) {
        int idx3d = idx;
        int idx5d = base + idx;
        Complex tmp1 = cuCmul(alpha1[idx5d], f_hat[idx3d]);
        Complex tmp2 = cuCmul(cuCconj(alpha1[idx5d]), f_hat[idx3d]);
        alpha1_times_f_hat[idx5d] = make_cuDoubleComplex(scale * tmp1.x, scale * tmp1.y);
        alpha2_times_f_hat[idx5d] = make_cuDoubleComplex(scale * tmp2.x, scale * tmp2.y);
    }                               
}
*/

// Kernel that computes alpha1_times_f_hat and alpha2_times_f_hat
// We map (r,s,i) to thread blocks and the loops over (j,k) are handled by the threads within each block
// Note: When launching this kernel, the grid size should be set to N_gl * N_spherical * Nvx
__global__ void compute_alpha_times_f_hat(Complex * alpha1_times_f_hat, Complex * alpha2_times_f_hat,
    const Complex * alpha1, const Complex * f_hat, 
    const int N_gl, const int N_spherical, 
    const int Nvx, const int Nvy, const int Nvz, 
    const double scale){

    // Compute the block index and backout the (r,s,i) indices
    int block_idx = blockIdx.x;
    int r = block_idx / (N_spherical * Nvx);
    int s = (block_idx / Nvx) % N_spherical;
    int i = block_idx % Nvx;
    const int base3d = i * Nvy * Nvz;
    const int base5d = ((r * N_spherical + s) * Nvx + i) * Nvy * Nvz;

    for (int idx = threadIdx.x; idx < Nvy * Nvz; idx += blockDim.x) {
        int idx3d = base3d + idx;
        int idx5d = base5d + idx;
        Complex tmp1 = cuCmul(alpha1[idx5d], f_hat[idx3d]);
        Complex tmp2 = cuCmul(cuCconj(alpha1[idx5d]), f_hat[idx3d]);
        alpha1_times_f_hat[idx5d] = cuCscale(tmp1, scale);
        alpha2_times_f_hat[idx5d] = cuCscale(tmp2, scale);
    }                               
}

// CUDA kernel to compute the Hadamard product of two Complex arrays
__global__ void hadamard_product(Complex * result, const Complex *x, const Complex *y, const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += num_threads) {
        result[i] = cuCmul(x[i], y[i]);
    }
}

/*
#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

#if defined(__CUDA_ARCH__)
  #pragma message("Compiling for __CUDA_ARCH__ = " STRINGIFY(__CUDA_ARCH__))
#else
  #pragma message("Compiling host code, __CUDA_ARCH__ not defined")
#endif
*/

// CUDA kernel to compute the tensor contraction giving Q_gain_hat using atomics
// We map (r,s,i) to thread blocks and the loops over (j,k) are handled by the threads within each block
// Note: When launching this kernel, the grid size should be set to N_gl * N_spherical * Nvx
__global__ void atomic_tensor_contraction(Complex * Q_gain_hat,
                                        const Complex * radial_term, const Complex * spherical_wts,
                                        const Complex * beta1, const Complex * transform_prod_hat,
                                        const int N_gl, const int N_spherical,
                                        const int Nvx, const int Nvy, const int Nvz,
                                        const double scale) {

    // Compute the block index and backout the (r,s,i) indices
    int block_idx = blockIdx.x;
    int r = block_idx / (N_spherical * Nvx);
    int s = (block_idx / Nvx) % N_spherical;
    int i = block_idx % Nvx;
    const int base3d = i * Nvy * Nvz;
    const int base4d = (r * Nvx + i) * Nvy * Nvz;
    const int base5d = ((r * N_spherical + s) * Nvx + i) * Nvy * Nvz;

    // Compute a weight which depends only on (r,s)
    Complex weight = cuCmul(radial_term[r], spherical_wts[s]);
    weight.x *= scale;
    weight.y *= scale;    

    double *Q_gain_hat_raw = reinterpret_cast<double*>(Q_gain_hat);

    for (int idx = threadIdx.x; idx < Nvy * Nvz; idx += blockDim.x) {
        int idx3d = base3d + idx;
        int idx4d = base4d + idx;
        int idx5d = base5d + idx;
        Complex tmp1 = cuCmul(beta1[idx4d], transform_prod_hat[idx5d]);
        Complex tmp2 = cuCmul(weight, tmp1);
 
        // Get pointers to real and imaginary parts and do an atomic add for each part
        atomicAdd(Q_gain_hat_raw + 2 * idx3d + 0, tmp2.x);
        atomicAdd(Q_gain_hat_raw + 2 * idx3d + 1, tmp2.y);
    }   
}

// CUDA kernel to compute the (scaled) elementwise product of a Complex array with a double array
__global__ void compute_beta2_times_f_hat(Complex * beta2_times_f_hat, const double * beta2, 
                                        const Complex * f_hat, const int N, const double scale){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += num_threads) {
        beta2_times_f_hat[i] = make_cuDoubleComplex(beta2[i] * scale * f_hat[i].x, 
                                                    beta2[i] * scale * f_hat[i].y);

    }
}

// CUDA kernel to compute the total Q = real(Q_gain) - real(Q_loss)
__global__ void compute_Q_total(double * Q, const Complex * Q_gain, 
                                const Complex * beta2_times_f_hat, const Complex * f, 
                                const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += num_threads) {
        Complex Q_loss = cuCmul(beta2_times_f_hat[i], f[i]);
        Q[i] = Q_gain[i].x - Q_loss.x;
    }

}



