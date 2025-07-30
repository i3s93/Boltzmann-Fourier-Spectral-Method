#include "CUDABoltzmannOperator.hpp"

std::string cufftGetErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default:
            return "Unknown CUFFT error (code = " + std::to_string(static_cast<int>(error)) + ")";
    }
}

// Initialize plans and arrays for the CUDA backend
void BoltzmannOperator<CUDA_Backend>::initialize() {

    int N_gl = gl_quadrature->getNumberOfPoints(); // Number of Gauss-Legendre quadrature points
    int N_spherical = spherical_quadrature->getNumberOfPoints(); // Number of spherical quadrature points
    int batch_size = N_gl * N_spherical; // Total number of grid_size batches
    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points    

    // Allocations for the device arrays used for the transforms
    // Consider using in-place transforms to save memory, along with r2c and c2r transforms (later)
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&f, grid_size * sizeof(cuDoubleComplex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&f_hat, grid_size * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha1_times_f, batch_size * grid_size * sizeof(cuDoubleComplex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha1_times_f_hat, batch_size * grid_size * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha2_times_f, batch_size * grid_size * sizeof(cuDoubleComplex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha2_times_f_hat, batch_size * grid_size * sizeof(cuDoubleComplex)));

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&beta2_times_f, grid_size * sizeof(cuDoubleComplex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&beta2_times_f_hat, grid_size * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&transform_prod, batch_size * grid_size * sizeof(cuDoubleComplex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&transform_prod_hat, batch_size * grid_size * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&Q_gain, grid_size * sizeof(cuDoubleComplex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&Q_gain_hat, grid_size * sizeof(cuDoubleComplex)) );

    // Allocate device arrays for the quadrature nodes and weights
    // We need these to be complex in order to use the cuTensorNet contractions
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&gl_wts, N_gl * sizeof(double)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&gl_nodes, N_gl * sizeof(double)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&spherical_wts, N_spherical * sizeof(double)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&sx, N_spherical * sizeof(double)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&sy, N_spherical * sizeof(double)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&sz, N_spherical * sizeof(double)) );

    // Allocate device arrays for the Fourier modes
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&lx, Nvx * sizeof(int)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&ly, Nvy * sizeof(int)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&lz, Nvz * sizeof(int)) );

    // Initialize the (host) arrays for the Fourier modes
    std::vector<int> lx_h, ly_h, lz_h; 
    lx_h.reserve(Nvx);
    ly_h.reserve(Nvy);
    lz_h.reserve(Nvz);
    
    for (int i = 0; i < Nvx / 2; ++i) lx_h.push_back(i);
    for (int i = -Nvx / 2; i < 0; ++i) lx_h.push_back(i);

    for (int j = 0; j < Nvy / 2; ++j) ly_h.push_back(j);
    for (int j = -Nvy / 2; j < 0; ++j) ly_h.push_back(j);

    for (int k = 0; k < Nvz / 2; ++k) lz_h.push_back(k);
    for (int k = -Nvz / 2; k < 0; ++k) lz_h.push_back(k);

    // We need to create single (3D) and batched (3D) plans for the transforms
    CUFFT_CALL( cufftPlan3d(&plan3d, Nvx, Nvy, Nvz, CUFFT_Z2Z) );

    int batched_rank = 3; // Each FFT is applied to a three-dimensional row-major array
    std::vector<int> batched_dims{Nvx, Nvy, Nvz}; // Dimensions of the arrays used in each transform
    int idist = grid_size; // Each array is separated by idist elements
    int odist = idist; // Each array is separated by odist elements
    int istride = 1; // Arrays are contiguous in memory
    int ostride = 1; // Arrays are contiguous in memory
    int *inembed = batched_dims.data(); // The array is not embedded in a larger array
    int *onembed = batched_dims.data(); // The array is not embedded in a larger array

    CUFFT_CALL( cufftPlanMany(&plan3d_batched, batched_rank, batched_dims.data(),
                inembed, istride, idist,
                onembed, ostride, odist,
                CUFFT_Z2Z, batch_size) );

    // Perform all host-to-device copies for the calculations
    HANDLE_CUDA_ERROR( cudaMemcpy(gl_wts, gl_quadrature->getWeights().data(), N_gl * sizeof(double), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(gl_nodes, gl_quadrature->getNodes().data(), N_gl * sizeof(double), cudaMemcpyHostToDevice) );

    HANDLE_CUDA_ERROR( cudaMemcpy(spherical_wts, spherical_quadrature->getWeights().data(), N_spherical * sizeof(double), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(sx, spherical_quadrature->getx().data(), N_spherical * sizeof(double), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(sy, spherical_quadrature->gety().data(), N_spherical * sizeof(double), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(sz, spherical_quadrature->getz().data(), N_spherical * sizeof(double), cudaMemcpyHostToDevice) );

    HANDLE_CUDA_ERROR( cudaMemcpy(lx, lx_h.data(), Nvx * sizeof(int), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(ly, ly_h.data(), Nvy * sizeof(int), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(lz, lz_h.data(), Nvz * sizeof(int), cudaMemcpyHostToDevice) );

}; // End of initialize


// Compute the collision operator using cuFFT and cuTensorNet
void BoltzmannOperator<CUDA_Backend>::computeCollision(double * Q, const double * f_in) {

    // Retrieve information from the quadrature objects
    int N_gl = gl_quadrature->getNumberOfPoints();
    int N_spherical = spherical_quadrature->getNumberOfPoints();

    int batch_size = N_gl * N_spherical; // Total number of grid_size batches
    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points
    double fft_scale = 1.0 / grid_size; // Scaling used to normalize transforms
    const int num_threads_per_block = 256; // Number of threads per CUDA block
    int num_blocks; // Number of CUDA blocks to launch (changes depending on the kernel)

    // Initialize the input as a complex array
    // We use half as many blocks to make each thread process roughly two elements
    num_blocks = std::max( int( grid_size / (2 * num_threads_per_block) ), 1 );
    copy_to_complex<<<num_blocks, num_threads_per_block>>>(f, f_in, grid_size);

    // Compute f_hat = fft(f)
    CUFFT_CALL( cufftExecZ2Z(plan3d, 
                reinterpret_cast<cufftDoubleComplex*>(f), 
                reinterpret_cast<cufftDoubleComplex*>(f_hat),
                CUFFT_FORWARD) );

    // Initialize the input as a complex array
    // We must launch EXACTLY one block per (r,s,i) triplet, i.e., "batch_size * Nvx" blocks
    num_blocks = batch_size * Nvx;
    compute_alpha_times_f_hat<<<num_blocks, num_threads_per_block>>>(alpha1_times_f_hat,
                                                                    alpha2_times_f_hat,
                                                                    f_hat,
                                                                    gl_nodes,
                                                                    lx, ly, lz,
                                                                    sx, sy, sz,
                                                                    N_gl, N_spherical, Nvx, Nvy, Nvz,
                                                                    pi, L, fft_scale);

    // Compute alpha1_times_f = ifft(alpha1_times_f_hat) and 
    // alpha2_times_f = ifft(alpha2_times_f_hat)
    CUFFT_CALL( cufftExecZ2Z(plan3d_batched, 
                reinterpret_cast<cufftDoubleComplex*>(alpha1_times_f_hat), 
                reinterpret_cast<cufftDoubleComplex*>(alpha1_times_f), 
                CUFFT_INVERSE) );

    CUFFT_CALL( cufftExecZ2Z(plan3d_batched,
                reinterpret_cast<cufftDoubleComplex*>(alpha2_times_f_hat), 
                reinterpret_cast<cufftDoubleComplex*>(alpha2_times_f), 
                CUFFT_INVERSE) );

    // Compute transform_prod = alpha1_times_f * alpha2_times_f
    // We use half as many blocks to make each thread process roughly two elements
    num_blocks = std::max( int( (batch_size * grid_size) / (2 * num_threads_per_block) ), 1 );
    hadamard_product<<<num_blocks, num_threads_per_block>>>(transform_prod, 
                                        alpha1_times_f, 
                                        alpha2_times_f, 
                                        batch_size * grid_size);

    // Compute transform_prod_hat = fft(transform_prod)
    CUFFT_CALL( cufftExecZ2Z(plan3d_batched, 
                reinterpret_cast<cufftDoubleComplex*>(transform_prod), 
                reinterpret_cast<cufftDoubleComplex*>(transform_prod_hat),
                CUFFT_FORWARD) );

    // Use tensor contraction over slices along modes 'r' and 's' to compute the following:
    // Q_gain_hat[i,j,k] = \sum_{r,s} radial_term[r] spherical_wts[s] beta1[r,i,j,k] transform_prod_hat[r,s,i,j,k]
    //
    // TO-DO: Overlap the initialization for Q_gain_hat with the copy from real to complex...
    HANDLE_CUDA_ERROR( cudaMemset(Q_gain_hat, 0.0, grid_size * sizeof(cuDoubleComplex)) ); // Initialize to zero

    // For now, just do the contraction using atomic adds
    num_blocks = batch_size * Nvx;
    atomic_tensor_contraction<<<num_blocks, num_threads_per_block >>>(Q_gain_hat,
        gl_nodes, gl_wts, spherical_wts, lx, ly, lz, transform_prod_hat,
        N_gl, N_spherical, Nvx, Nvy, Nvz, 
        gamma, b_gamma, pi, L, fft_scale);

    // Compute beta2_times_f_hat = beta2 * f_hat
    num_blocks = std::max( int( grid_size / (2 * num_threads_per_block) ), 1 );
    compute_beta2_times_f_hat<<<num_blocks, num_threads_per_block>>>(beta2_times_f_hat, 
        f_hat, gl_wts, gl_nodes, lx, ly, lz,
        N_gl, Nvx, Nvy, Nvz,
        gamma, b_gamma, pi, L, fft_scale);

    // Note: The two transforms below can be run in different streams if needed (do this later...)

    // Compute Q_gain = ifft(Q_gain_hat)
    CUFFT_CALL( cufftExecZ2Z(plan3d, 
                reinterpret_cast<cufftDoubleComplex*>(Q_gain_hat), 
                reinterpret_cast<cufftDoubleComplex*>(Q_gain),
                CUFFT_INVERSE) );

    // Compute beta2_times_f = ifft(beta2_times_f_hat)
    CUFFT_CALL( cufftExecZ2Z(plan3d, 
                reinterpret_cast<cufftDoubleComplex*>(beta2_times_f_hat), 
                reinterpret_cast<cufftDoubleComplex*>(beta2_times_f),
                CUFFT_INVERSE) );

    // Compute Q = real(Q_gain) - real(Q_loss)
    num_blocks = std::max( int( grid_size / (2 * num_threads_per_block) ), 1 );
    compute_Q_total<<<num_blocks, num_threads_per_block>>>(Q, Q_gain, beta2_times_f, f, grid_size);

    cudaDeviceSynchronize(); // Ensure all device kernels have completed before returning

}; // End of computeCollision


// Class destructor to clean up CUDA resources
BoltzmannOperator<CUDA_Backend>::~BoltzmannOperator() {

    // Clean up cuFFT plans
    CUFFT_CALL( cufftDestroy(plan3d) );
    CUFFT_CALL( cufftDestroy(plan3d_batched) );

    // Free allocated arrays for the transforms and weights
    HANDLE_CUDA_ERROR( cudaFree(f) );
    HANDLE_CUDA_ERROR( cudaFree(f_hat) );

    HANDLE_CUDA_ERROR( cudaFree(alpha1_times_f) );
    HANDLE_CUDA_ERROR( cudaFree(alpha1_times_f_hat) );
    
    HANDLE_CUDA_ERROR( cudaFree(alpha2_times_f) );
    HANDLE_CUDA_ERROR( cudaFree(alpha2_times_f_hat) );
    
    HANDLE_CUDA_ERROR( cudaFree(beta2_times_f) );
    HANDLE_CUDA_ERROR( cudaFree(beta2_times_f_hat) );
    
    HANDLE_CUDA_ERROR( cudaFree(transform_prod) );
    HANDLE_CUDA_ERROR( cudaFree(transform_prod_hat) );
    
    HANDLE_CUDA_ERROR( cudaFree(Q_gain_hat) );
    HANDLE_CUDA_ERROR( cudaFree(Q_gain) );

    HANDLE_CUDA_ERROR( cudaFree(gl_wts) );
    HANDLE_CUDA_ERROR( cudaFree(gl_nodes) );
    HANDLE_CUDA_ERROR( cudaFree(spherical_wts) );
    
    HANDLE_CUDA_ERROR( cudaFree(sx) );
    HANDLE_CUDA_ERROR( cudaFree(sy) );
    HANDLE_CUDA_ERROR( cudaFree(sz) );

    HANDLE_CUDA_ERROR( cudaFree(lx) );
    HANDLE_CUDA_ERROR( cudaFree(ly) );
    HANDLE_CUDA_ERROR( cudaFree(lz) );

}; // End of destructor
