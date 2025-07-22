#include "BoltzmannCUDAKernels.hpp"


// CUDA kernel to copy a double array to a complex array
__global__ void copy_to_complex(
    cuDoubleComplex * __restrict__ result, 
    const double * __restrict__ input, 
    const int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int idx = tid; idx < N; idx += num_threads) {
        result[idx].x = input[idx];
        result[idx].y = 0.0;
    }
}

// Kernel that computes alpha1_times_f_hat and alpha2_times_f_hat
// We map (r,s,i) to thread blocks and the loops over (j,k) are handled by the threads within each block
// Note: When launching this kernel, the grid size must be N_gl * N_spherical * Nvx
__global__ void compute_alpha_times_f_hat(
    cuDoubleComplex * __restrict__ alpha1_times_f_hat,
    cuDoubleComplex * __restrict__ alpha2_times_f_hat,
    const cuDoubleComplex * __restrict__ f_hat,
    const double * __restrict__ gl_nodes,
    const int * __restrict__ lx, const int * __restrict__ ly, const int * __restrict__ lz,
    const double * __restrict__ sx, const double * __restrict__ sy, const double * __restrict__ sz,
    const int N_gl, const int N_spherical, const int Nvx, const int Nvy, const int Nvz,
    const double pi, const double L, const double fft_scale){

    int block_idx = blockIdx.x;
    int r = block_idx / (N_spherical * Nvx);
    int s = (block_idx / Nvx) % N_spherical;
    int i = block_idx % Nvx;

    if (r >= N_gl || s >= N_spherical || i >= Nvx) return;

    const int base3d = i * Nvy * Nvz;
    const int base5d = ((r * N_spherical + s) * Nvx + i) * Nvy * Nvz;

    for (int idx = threadIdx.x; idx < Nvy * Nvz; idx += blockDim.x) {
        int idx3d = base3d + idx;
        int idx5d = base5d + idx;
        int j = idx / Nvz;
        int k = idx % Nvz;

        double l_dot_sigma = lx[i]*sx[s] + ly[j]*sy[s] + lz[k]*sz[s];
        double tmp = -(pi/(2*L)) * gl_nodes[r] * l_dot_sigma;
        double a_re = cos(tmp);
        double a_im = sin(tmp);
        cuDoubleComplex b = f_hat[idx3d];

        alpha1_times_f_hat[idx5d].x = fft_scale * (a_re * b.x - a_im * b.y);
        alpha1_times_f_hat[idx5d].y = fft_scale * (a_re * b.y + a_im * b.x);

        alpha2_times_f_hat[idx5d].x = fft_scale * (a_re * b.x + a_im * b.y);
        alpha2_times_f_hat[idx5d].y = fft_scale * (a_re * b.y - a_im * b.x);
    }
}

// CUDA kernel to compute the Hadamard product of two complex arrays
__global__ void hadamard_product(
    cuDoubleComplex * __restrict__ result, 
    const cuDoubleComplex *  __restrict__ x, 
    const cuDoubleComplex *  __restrict__ y, 
    const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += num_threads) {
        result[i] = cuCmul(x[i], y[i]);
    }
}

// CUDA kernel to compute the tensor contraction giving Q_gain_hat using atomics
// We map (r,s,i) to thread blocks and the loops over (j,k) are handled by the threads within each block
// Note: When launching this kernel, the grid size should be set to N_gl * N_spherical * Nvx
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
    const double pi, const double L, const double fft_scale){

    int block_idx = blockIdx.x;
    int r = block_idx / (N_spherical * Nvx);
    int s = (block_idx / Nvx) % N_spherical;
    int i = block_idx % Nvx;

    if (r >= N_gl || s >= N_spherical || i >= Nvx) return; // Is this necessary?

    const int base3d = i * Nvy * Nvz;
    const int base5d = ((r * N_spherical + s) * Nvx + i) * Nvy * Nvz;

    double weight = fft_scale * gl_wts[r] * spherical_wts[s] * pow(gl_nodes[r], gamma + 2);

    double *Q_gain_hat_ptr = reinterpret_cast<double*>(Q_gain_hat);

    for (int idx = threadIdx.x; idx < Nvy * Nvz; idx += blockDim.x) {
        int idx3d = base3d + idx;
        int idx5d = base5d + idx;
        int j = idx / Nvz;
        int k = idx % Nvz;

        double norm_l = sqrt(lx[i]*lx[i] + ly[j]*ly[j] + lz[k]*lz[k]);
        double beta1 = weight * 4 * pi * b_gamma * sincc(pi * gl_nodes[r] * norm_l / (2 * L));

        cuDoubleComplex tmp;
        tmp.x = beta1 * transform_prod_hat[idx5d].x;
        tmp.y = beta1 * transform_prod_hat[idx5d].y;

        atomicAdd(Q_gain_hat_ptr + 2 * idx3d + 0, tmp.x);
        atomicAdd(Q_gain_hat_ptr + 2 * idx3d + 1, tmp.y);
    }
}

// CUDA kernel to compute the (scaled) elementwise product of a complex array with a double array
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
    const double fft_scale){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nvx * Nvy * Nvz;
    int num_threads = blockDim.x * gridDim.x;

    // TO-DO: Consider loading in shared memory: gl_nodes, and gl_wts 

    for (int idx = tid; idx < total; idx += num_threads) {
        int i = idx / (Nvy * Nvz);
        int j = (idx / Nvz) % Nvy;
        int k = idx % Nvz;

        double norm_l = sqrt(lx[i]*lx[i] + ly[j]*ly[j] + lz[k]*lz[k]);

        double beta2 = 0;
        for (int r = 0; r < N_gl; ++r) {
            beta2 += 16 * pi * pi * b_gamma * gl_wts[r] * pow(gl_nodes[r], gamma+2) * sincc(pi * gl_nodes[r] * norm_l / L);
        }

        beta2_times_f_hat[idx] = make_cuDoubleComplex(fft_scale * beta2 * f_hat[idx].x, 
                                                      fft_scale * beta2 * f_hat[idx].y);
    }
}

// CUDA kernel to compute the total Q = real(Q_gain) - real(Q_loss)
__global__ void compute_Q_total(
    double * __restrict__ Q, 
    const cuDoubleComplex * __restrict__ Q_gain, 
    const cuDoubleComplex * __restrict__ beta2_times_f_hat, 
    const cuDoubleComplex * __restrict__ f, 
    const int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int idx = tid; idx < N; idx += num_threads) {
        cuDoubleComplex Q_loss = cuCmul(beta2_times_f_hat[idx], f[idx]);
        Q[idx] = Q_gain[idx].x - Q_loss.x;
    }

}



