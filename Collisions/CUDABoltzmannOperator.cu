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

    // Allocations for the device arrays used the operator
    // Note: alpha2 = conj(alpha1) so we don't store it
    // Note: beta1 is real but we store it as complex to use cuTensorNet contractions
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha1, batch_size * grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&beta1, N_gl * grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&beta2, grid_size * sizeof(double)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&f, grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&f_hat, grid_size * sizeof(Complex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha1_times_f, batch_size * grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha1_times_f_hat, batch_size * grid_size * sizeof(Complex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha2_times_f, batch_size * grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&alpha2_times_f_hat, batch_size * grid_size * sizeof(Complex)));

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&beta2_times_f, grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&beta2_times_f_hat, grid_size * sizeof(Complex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&transform_prod, batch_size * grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&transform_prod_hat, batch_size * grid_size * sizeof(Complex)) );

    HANDLE_CUDA_ERROR( cudaMalloc((void **)&Q_gain, grid_size * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&Q_gain_hat, grid_size * sizeof(Complex)) );

    // Allocate device arrays for the quadrature nodes and weights
    // We need these to be complex in order to use the cuTensorNet contractions
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&radial_term, N_gl * sizeof(Complex)) );
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&spherical_wts, N_spherical * sizeof(Complex)) );

    // Initialize the arrays for the Fourier modes
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
/*
    // Now we need to initialize a context to the cuTensorNet library
    HANDLE_ERROR( cutensornetCreate(&cutensornet_handle) );

    // Next we provide descriptions of the tensors used in the contraction
    // By providing modes, extents, and strides of each tensor in the operation:
    // Q_gain_hat[i,j,k] = \sum_{r,s} radial_term[r] * spherical_wts[s] *
    //                     gl_nodes[r]^(gamma+2) * beta1[r,i,j,k] * transform_prod_hat[r,s,i,j,k]
    std::vector<int32_t> modes_radial_term{'r'}; 
    std::vector<int32_t> modes_spherical_wts{'s'}; 
    std::vector<int32_t> modes_beta1{'r', 'i', 'j', 'k'}; 
    std::vector<int32_t> modes_transform_prod_hat{'r', 's', 'i', 'j', 'k'}; 
    std::vector<int32_t> modes_Q_gain_hat{'i', 'j', 'k'}; 

    std::vector<int64_t> extents_radial_term{N_gl}; 
    std::vector<int64_t> extents_spherical_wts{N_spherical};
    std::vector<int64_t> extents_beta1{N_gl, Nvx, Nvy, Nvz}; 
    std::vector<int64_t> extents_transform_prod_hat{N_gl, N_spherical, Nvx, Nvy, Nvz}; 
    std::vector<int64_t> extents_Q_gain_hat{Nvx, Nvy, Nvz};
    
    std::vector<int64_t> strides_radial_term{1}; 
    std::vector<int64_t> strides_spherical_wts{1};
    std::vector<int64_t> strides_beta1{Nvx*Nvy*Nvz, Nvy*Nvz, Nvz, 1}; 
    std::vector<int64_t> strides_transform_prod_hat{N_spherical*Nvx*Nvy*Nvz, Nvx*Nvy*Nvz, Nvy*Nvz, Nvz, 1}; 
    std::vector<int64_t> strides_Q_gain_hat{Nvx*Nvy*Nvz, Nvy*Nvz, Nvz, 1}; 

    int32_t numInputs = 4;
    int32_t const numModesIn[] = {1, 1, 4, 5};

    int32_t* modesIn[] = {modes_radial_term.data(), modes_spherical_wts.data(), 
        modes_beta1.data(), modes_transform_prod_hat.data()};
    
    int64_t* extentsIn[] = {extents_radial_term.data(), extents_spherical_wts.data(), 
                                    extents_beta1.data(), extents_transform_prod_hat.data()};

    int64_t* stridesIn[] = {strides_radial_term.data(), strides_spherical_wts.data(), 
                                    strides_beta1.data(), strides_transform_prod_hat.data()};

    int32_t numModesOut = 3;

    // Finalize the setup for the network descriptor
    HANDLE_ERROR( cutensornetCreateNetworkDescriptor(cutensornet_handle,
        numInputs, numModesIn, extentsIn, stridesIn, modesIn, nullptr,
        numModesOut, extents_Q_gain_hat.data(), strides_Q_gain_hat.data(), modes_Q_gain_hat.data(),
        CUDA_C_64F, CUTENSORNET_COMPUTE_64F,
        &descNet) );

    // Next we setup the optimizer config structure which hold the hyperparameters for the contraction
    // We also adjust the number of hyper-samples to use during the optimization
    int32_t num_hypersamples = 8;
    HANDLE_ERROR( cutensornetCreateContractionOptimizerConfig(cutensornet_handle, &optimizerConfig) );
    HANDLE_ERROR( cutensornetContractionOptimizerConfigSetAttribute(cutensornet_handle,
                     optimizerConfig,
                     CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
                     &num_hypersamples,
                     sizeof(num_hypersamples)) );

    // Allocate resources for the optimizer and find and optimized contraction path
    // This sets a hard limit on the workspace size used by the optimizer using available memory
    size_t freeMem, totalMem;
    HANDLE_CUDA_ERROR( cudaMemGetInfo(&freeMem, &totalMem) );
    uint64_t workspaceLimit = (uint64_t)((double)freeMem * 0.9);

    HANDLE_ERROR( cutensornetCreateContractionOptimizerInfo(cutensornet_handle, descNet, &optimizerInfo));
    HANDLE_ERROR( cutensornetContractionOptimize(cutensornet_handle, descNet, optimizerConfig, 
                                                 workspaceLimit, optimizerInfo) );

    // Create workspace descriptor, allocate workspace, and then set it
    int64_t requiredWorkspaceSize = 0;
    HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(cutensornet_handle, &workDesc) );
    HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(cutensornet_handle, workDesc,
                                                    CUTENSORNET_WORKSIZE_PREF_MIN,
                                                    CUTENSORNET_MEMSPACE_DEVICE,
                                                    CUTENSORNET_WORKSPACE_SCRATCH,
                                                    &requiredWorkspaceSize) );

    // Allocate the workspace on the device
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&workspace, requiredWorkspaceSize) );

    HANDLE_ERROR( cutensornetWorkspaceSetMemory(cutensornet_handle, workDesc,
                                                CUTENSORNET_MEMSPACE_DEVICE,
                                                CUTENSORNET_WORKSPACE_SCRATCH,
                                                workspace,
                                                requiredWorkspaceSize) );

    // Setup the contraction plan
    HANDLE_ERROR( cutensornetCreateContractionPlan(cutensornet_handle, descNet, optimizerInfo, 
                                                workDesc, &contraction_plan) );

    // Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel for each pairwise contraction
    HANDLE_ERROR( cutensornetCreateContractionAutotunePreference(cutensornet_handle, &autotunePref) );

    const int32_t numAutotuningIterations = 5;
    HANDLE_ERROR( cutensornetContractionAutotunePreferenceSetAttribute(
                            cutensornet_handle,
                            autotunePref,
                            CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                            &numAutotuningIterations,
                            sizeof(numAutotuningIterations)) );
*/
}; // End of initialize

// Precompute the transform weights alpha1, beta1, beta2
void BoltzmannOperator<CUDA_Backend>::precomputeTransformWeights() {

    int N_gl = gl_quadrature->getNumberOfPoints(); // Number of Gauss-Legendre quadrature points
    int N_spherical = spherical_quadrature->getNumberOfPoints(); // Number of spherical quadrature points
    int batch_size = N_gl * N_spherical; // Total number of grid_size batches
    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points

    // Create some host arrays to store the precomputed weights
    std::vector<Complex> alpha1_h(batch_size * grid_size);
    std::vector<Complex> beta1_h(N_gl * grid_size);
    std::vector<double> beta2_h(grid_size);

    // Extract the quadrature weights and nodes (using shallow copies)
    const std::vector<double>& gl_wts_h = gl_quadrature->getWeights();
    const std::vector<double>& gl_nodes_h = gl_quadrature->getNodes();
    const std::vector<double>& spherical_wts_h = spherical_quadrature->getWeights();
    const std::vector<double>& sx_h = spherical_quadrature->getx();
    const std::vector<double>& sy_h = spherical_quadrature->gety();
    const std::vector<double>& sz_h = spherical_quadrature->getz();

    // Allocations for the radial term and spherical weights (complex)
    std::vector<Complex> radial_term_h(N_gl);
    std::vector<Complex> complex_spherical_wts_h(N_spherical);

    if (lx_h.empty() || ly_h.empty() || lz_h.empty()) {
        throw std::runtime_error("lx, ly, or lz is not initialized");
    }

    if (sx_h.empty() || sy_h.empty() || sz_h.empty()) {
        throw std::runtime_error("sx, sy, or sz is not initialized");
    }
    
    #pragma omp parallel
    {

    // Compute the complex transform weights alpha1
    // Note that we do not compute alpha2 since it is the conjugate of alpha1
    #pragma omp for collapse(4) nowait
    for (int r = 0; r < N_gl; ++r){
        for (int s = 0; s < N_spherical; ++s){
            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nvz; ++k){
                        int idx5 = ((((r) * N_spherical + s) * Nvx + i) * Nvy + j) * Nvz + k;
                        double l_dot_sigma = lx_h[i]*sx_h[s] + ly_h[j]*sy_h[s] + lz_h[k]*sz_h[s];
                        Complex arg = make_cuDoubleComplex(0.0, -(pi / (2 * L)) * gl_nodes_h[r] * l_dot_sigma);                   
                        alpha1_h[idx5] = cuCexp(arg);
                    }
                }
            }
        }
    }

    // Compute the real transform weights beta1
    // Note: We convert beta1 to complex to use cuTensorNet contractions
    #pragma omp for simd collapse(3) nowait
    for (int r = 0; r < N_gl; ++r){
        for (int i = 0; i < Nvx; ++i){
            for (int j = 0; j < Nvy; ++j){
                #pragma omp simd
                for (int k = 0; k < Nvz; ++k){
                    int idx4 = (((r * Nvx + i) * Nvy + j) * Nvz + k);
                    double norm_l = std::sqrt(lx_h[i]*lx_h[i] + ly_h[j]*ly_h[j] + lz_h[k]*lz_h[k]);
                    beta1_h[idx4] = make_cuDoubleComplex( 4*pi*b_gamma*sincc(pi*gl_nodes_h[r]*norm_l/(2*L)), 0.0 );
                }
            }
        }
    }

    // Compute the real transform weights beta2
    #pragma omp for collapse(3) nowait
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            for (int k = 0; k < Nvz; ++k){

                int idx3 = (i * Nvy + j) * Nvz + k;
                double tmp = 0.0;
                double norm_l = std::sqrt(lx_h[i]*lx_h[i] + ly_h[j]*ly_h[j] + lz_h[k]*lz_h[k]);

                #pragma omp simd reduction(+:tmp)
                for (int r = 0; r < N_gl; ++r){
                    tmp += 16*pi*pi*b_gamma*gl_wts_h[r]*std::pow(gl_nodes_h[r], gamma+2)*sincc(pi*gl_nodes_h[r]*norm_l/L);
                }

                beta2_h[idx3] = tmp;

            }
        }
    }

    #pragma omp single nowait
    for (int r = 0; r < N_gl; ++r) {
        radial_term_h[r] = make_cuDoubleComplex(gl_wts_h[r] * std::pow(gl_nodes_h[r], gamma + 2), 0.0);
    }

    #pragma omp single nowait
    for (int s = 0; s < N_spherical; ++s) {
        complex_spherical_wts_h[s] = make_cuDoubleComplex(spherical_wts_h[s], 0.0);
    }

    } // End of parallel region

    // Transfer precomputed data from the host to the device
    HANDLE_CUDA_ERROR( cudaMemcpy(alpha1, alpha1_h.data(), batch_size * grid_size * sizeof(Complex), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(beta1, beta1_h.data(), N_gl * grid_size * sizeof(Complex), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(beta2, beta2_h.data(), grid_size * sizeof(double), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(radial_term, radial_term_h.data(), N_gl * sizeof(Complex), cudaMemcpyHostToDevice) );
    HANDLE_CUDA_ERROR( cudaMemcpy(spherical_wts, complex_spherical_wts_h.data(), N_spherical * sizeof(Complex), cudaMemcpyHostToDevice) );

}; // End of precomputeTransformWeights

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
    // We must launch exactly one block per (r,s,i) triplet
    num_blocks = batch_size * Nvx;
    compute_alpha_times_f_hat<<<num_blocks, num_threads_per_block>>>(alpha1_times_f_hat, 
                                    alpha2_times_f_hat, alpha1, f_hat, 
                                    N_gl, N_spherical, Nvx, Nvy, Nvz, fft_scale);

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
    HANDLE_CUDA_ERROR( cudaMemset(Q_gain_hat, 0.0, grid_size * sizeof(Complex)) ); // Initialize to zero

    // For now, just do the contraction using atomic adds
    num_blocks = batch_size * Nvx;
    atomic_tensor_contraction<<<num_blocks, num_threads_per_block >>>(Q_gain_hat,
                                        radial_term, spherical_wts, beta1, transform_prod_hat,
                                        N_gl, N_spherical, Nvx, Nvy, Nvz, fft_scale);

/*
    // The input arrays used in the contractions need to be stored in a container of type void for generality
    const void* rawDataIn[] = {radial_term, spherical_wts, beta1, transform_prod_hat};

    HANDLE_ERROR( cutensornetContractSlices(cutensornet_handle,
                   contraction_plan,
                   rawDataIn, // Array of (device) pointers to input tensors
                   Q_gain_hat, // Output tensor
                   1, // int32_t accumulateOutput = 1 means we accumulate the output
                   workDesc,
                   nullptr, // nullptr means we contract over all slices instead of specifying a sliceGroup object
                   0) ); // Use the default stream
*/

    // Compute beta2_times_f_hat = beta2 * f_hat
    num_blocks = std::max( int( grid_size / (2 * num_threads_per_block) ), 1 );
    compute_beta2_times_f_hat<<<num_blocks, num_threads_per_block>>>(beta2_times_f_hat, beta2, f_hat, grid_size, fft_scale);

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
    compute_Q_total<<<num_blocks, num_threads_per_block>>>(Q, Q_gain, beta2_times_f_hat, f_hat, grid_size);

}; // End of computeCollision


// Class destructor to clean up CUDA resources
BoltzmannOperator<CUDA_Backend>::~BoltzmannOperator() {

    // Clean up cuFFT plans
    CUFFT_CALL( cufftDestroy(plan3d) );
    CUFFT_CALL( cufftDestroy(plan3d_batched) );

    // Free the contraction plans
/*
    HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
    HANDLE_ERROR( cutensornetDestroyContractionAutotunePreference(autotunePref) );
    HANDLE_ERROR( cutensornetDestroyContractionPlan(contraction_plan) );
    HANDLE_ERROR( cutensornetDestroyContractionOptimizerConfig(optimizerConfig) ); 
    HANDLE_ERROR( cutensornetDestroyContractionOptimizerInfo(optimizerInfo) );   
    HANDLE_ERROR( cutensornetDestroyNetworkDescriptor(descNet) );
    HANDLE_ERROR( cutensornetDestroy(cutensornet_handle) );
*/
    // Free allocated arrays for the transforms and weights
    HANDLE_CUDA_ERROR( cudaFree(alpha1) );
    HANDLE_CUDA_ERROR( cudaFree(beta1) );
    HANDLE_CUDA_ERROR( cudaFree(beta2) );

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

    HANDLE_CUDA_ERROR( cudaFree(radial_term) );
    HANDLE_CUDA_ERROR( cudaFree(spherical_wts) );
    //HANDLE_CUDA_ERROR( cudaFree(workspace) );

}; // End of destructor
