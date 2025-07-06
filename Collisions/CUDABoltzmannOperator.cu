#include "CUDABoltzmannOperator.hpp"


// Initialize plans and arrays for the FFTW backend
void BoltzmannOperator<CUDA_Backend>::initialize() {

    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points
    int N_gl = gl_quadrature->getNumberOfPoints(); // Number of Gauss-Legendre quadrature points
    int N_spherical = spherical_quadrature->getNumberOfPoints(); // Number of spherical quadrature points
    int batch_size = N_gl * N_spherical; // Total number of grid_size batches

    // Allocations for the device arrays used the operator
    // Note: alpha2 = conj(alpha1) so we don't store it
    cudaMalloc((void **)&alpha1, batch_size * grid_size * sizeof(Complex));
    cudaMalloc((void **)&beta1, N_gl * grid_size * sizeof(double));
    cudaMalloc((void **)&beta2, grid_size * sizeof(double));

    cudaMalloc((void **)&f, grid_size * sizeof(Complex));
    cudaMalloc((void **)&f_hat, grid_size * sizeof(Complex));

    cudaMalloc((void **)&alpha1_times_f, batch_size * grid_size * sizeof(Complex));
    cudaMalloc((void **)&alpha1_times_f_hat, batch_size * grid_size * sizeof(Complex));

    cudaMalloc((void **)&alpha2_times_f, batch_size * grid_size * sizeof(Complex));
    cudaMalloc((void **)&alpha2_times_f_hat, batch_size * grid_size * sizeof(Complex));

    cudaMalloc((void **)&beta2_times_f, grid_size * sizeof(Complex));
    cudaMalloc((void **)&beta2_times_f_hat, grid_size * sizeof(Complex));

    cudaMalloc((void **)&transform_prod, batch_size * grid_size * sizeof(Complex));
    cudaMalloc((void **)&transform_prod_hat, batch_size * grid_size * sizeof(Complex));

    cudaMalloc((void **)&Q_gain, grid_size * sizeof(Complex));
    cudaMalloc((void **)&Q_gain_hat, grid_size * sizeof(Complex));

    // Allocate device arrays for the quadrature nodes and weights
    cudaMalloc((void **)&gl_nodes, N_gl * sizeof(double));
    cudaMalloc((void **)&gl_wts, N_gl * sizeof(double));
    cudaMalloc((void **)&spherical_wts, N_spherical * sizeof(double));

    // Initialize the arrays for the Fourier modes
    lx.reserve(Nvx);
    ly.reserve(Nvy);
    lz.reserve(Nvz);
    
    for (int i = 0; i < Nvx / 2; ++i) lx.push_back(i);
    for (int i = -Nvx / 2; i < 0; ++i) lx.push_back(i);

    for (int j = 0; j < Nvy / 2; ++j) ly.push_back(j);
    for (int j = -Nvy / 2; j < 0; ++j) ly.push_back(j);

    for (int k = 0; k < Nvz / 2; ++k) lz.push_back(k);
    for (int k = -Nvz / 2; k < 0; ++k) lz.push_back(k);

    // We need to create single (3D) and batched (3D) plans for the transforms
    cufftPlan3d(&plan3d, Nvx, Nvy, Nvz, CUFFT_Z2Z);

    int batched_rank = 3; // Each FFT is applied to a three-dimensional row-major array
    int batched_dims[] = {Nvx, Nvy, Nvz}; // Dimensions of the arrays used in each transform
    int idist = Nvx*Nvy*Nvz; // Each array is separated by idist elements
    int odist = idist; // Each array is separated by odist elements
    int istride = 1; // Arrays are contiguous in memory
    int ostride = 1; // Arrays are contiguous in memory
    int *inembed = batched_dims; // The array is not embedded in a larger array
    int *onembed = batched_dims; // The array is not embedded in a larger array

    cufftPlanMany(&plan3d_batched, batched_rank, batched_dims,
                inembed, istride, idist,
                onembed, ostride, odist,
                CUFFT_Z2Z, batch_size);

    // Now we need to initialize a context to the cuTensorNet library
    cutensornetCreate(&cutensornet_handle);

    // Next we provide descriptions of the tensors used in the contraction
    // By providing modes, extents, and strides of each tensor in the operation:
    // Q_gain_hat[i,j,k] = \sum_{r,s} gl_wts[r] * spherical_wts[s] *
    //                     gl_nodes[r]^(gamma+2) * beta1[r,i,j,k] * transform_prod_hat[r,s,i,j,k]
    std::vector<int32_t> modes_gl_wts{'r'}; 
    std::vector<int32_t> modes_spherical_wts{'s'}; 
    std::vector<int32_t> modes_beta1{'r', 'i', 'j', 'k'}; 
    std::vector<int32_t> modes_transform_prod_hat{'r', 's', 'i', 'j', 'k'}; 
    std::vector<int32_t> modes_Q_gain_hat{'i', 'j', 'k'}; 

    std::vector<int64_t> extents_gl_wts{N_gl}; 
    std::vector<int64_t> extents_spherical_wts{N_spherical};
    std::vector<int64_t> extents_beta1{N_gl, Nvx, Nvy, Nvz}; 
    std::vector<int64_t> extents_transform_prod_hat{N_gl, N_spherical, Nvx, Nvy, Nvz}; 
    std::vector<int64_t> extents_Q_gain_hat{Nvx, Nvy, Nvz};
    
    std::vector<int64_t> strides_gl_wts{1}; 
    std::vector<int64_t> strides_spherical_wts{1};
    std::vector<int64_t> strides_beta1{Nvx*Nvy*Nvz, Nvy*Nvz, Nvz, 1}; 
    std::vector<int64_t> strides_transform_prod_hat{N_spherical*Nvx*Nvy*Nvz, Nvx*Nvy*Nvz, Nvy*Nvz, Nvz, 1}; 
    std::vector<int64_t> strides_Q_gain_hat{Nvx*Nvy*Nvz, Nvy*Nvz, Nvz, 1}; 

    int32_t numInputs = 4;
    int32_t const numModesIn[] = {1, 1, 4, 5};
    const int32_t modesIn[] = {modes_gl_wts.data(), modes_spherical_wts.data(), 
        modes_beta1.data(), modes_transform_prod_hat.data()};
    const int64_t * extentsIn[] = {extents_gl_wts.data(), extents_spherical_wts.data(), 
                                    extents_beta1.data(), extents_transform_prod_hat.data()};
    const int64_t * stridesIn[] = {strides_gl_wts.data(), strides_spherical_wts.data(), 
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

    // Create workspace descriptor, allocate workspace, and then set it.
    requiredWorkspaceSize = 0;
    HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(cutensornet_handle, &workDesc) );
    HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle, workDesc,
                                                    CUTENSORNET_WORKSIZE_PREF_MIN,
                                                    CUTENSORNET_MEMSPACE_DEVICE,
                                                    CUTENSORNET_WORKSPACE_SCRATCH,
                                                    &requiredWorkspaceSize) );

    // Allocate the workspace on the device
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&workspace, requiredWorkspaceSize) );

    HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle, workDesc,
                                                CUTENSORNET_MEMSPACE_DEVICE,
                                                CUTENSORNET_WORKSPACE_SCRATCH,
                                                workspace,
                                                requiredWorkspaceSize) );

    // Setup the contraction plan
    HANDLE_ERROR( cutensornetCreateContractionPlan(cutensornet_handle, descNet, optimizerInfo, 
                                                   workspaceSize, &contraction_plan) );

   // Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel for each pairwise contraction
  HANDLE_ERROR( cutensornetCreateContractionAutotunePreference(handle,
                                                     &autotunePref) );

  const int_32_t numAutotuningIterations = 5;
  HANDLE_ERROR( cutensornetContractionAutotunePreferenceSetAttribute(
                          handle,
                          autotunePref,
                          CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                          &numAutotuningIterations,
                          sizeof(numAutotuningIterations)) );

}; // End of initialize

// Precompute the transform weights alpha1, beta1, beta2
void BoltzmannOperator<CUDA_Backend>::precomputeTransformWeights() {

    int N_gl = gl_quadrature->getNumberOfPoints(); // Number of Gauss-Legendre quadrature points
    int N_spherical = spherical_quadrature->getNumberOfPoints(); // Number of spherical quadrature points
    int batch_size = N_gl * N_spherical; // Total number of grid_size batches

    // Create some host arrays to store the precomputed weights
    std::vector<Complex> alpha1_h(batch_size * Nvx * Nvy * Nvz);
    std::vector<double> beta1_h(N_gl * Nvx * Nvy * Nvz);
    std::vector<double> beta2_h(Nvx * Nvy * Nvz);

    // Extract the quadrature weights and nodes (using shallow copies)
    const std::vector<double>& gl_wts_h = gl_quadrature->getWeights();
    const std::vector<double>& gl_nodes_h = gl_quadrature->getNodes();
    const std::vector<double>& sx_h = spherical_quadrature->getx();
    const std::vector<double>& sy_h = spherical_quadrature->gety();
    const std::vector<double>& sz_h = spherical_quadrature->getz();

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
    //
    // TO-DO: Modify this loop to use my version of Complex rather than std::complex 
    #pragma omp for collapse(4)
    for (int r = 0; r < N_gl; ++r){
        for (int s = 0; s < N_spherical; ++s){
            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nvz; ++k){
                        int idx5 = ((((r) * N_spherical + s) * Nvx + i) * Nvy + j) * Nvz + k;
                        double l_dot_sigma = lx_h[i]*sx_h[s] + ly_h[j]*sy_h[s] + lz_h[k]*sz_h[s];
                        alpha1_h[idx5] = std::exp(std::complex<double>(0,-(pi/(2*L))*gl_nodes_h[r]*l_dot_sigma));                   
                    }
                }
            }
        }
    }

    // Compute the real transform weights beta1
    #pragma omp for simd collapse(3)
    for (int r = 0; r < N_gl; ++r){
        for (int i = 0; i < Nvx; ++i){
            for (int j = 0; j < Nvy; ++j){
                #pragma omp simd
                for (int k = 0; k < Nvz; ++k){
                    int idx4 = (((r * Nvx + i) * Nvy + j) * Nvz + k);
                    double norm_l = std::sqrt(lx_h[i]*lx_h[i] + ly_h[j]*ly_h[j] + lz_h[k]*lz_h[k]);
                    beta1_h[idx4] = 4*pi*b_gamma*sincc(pi*gl_nodes_h[r]*norm_l/(2*L));
                }
            }
        }
    }

    // Compute the real transform weights beta2
    #pragma omp for collapse(3)
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            for (int k = 0; k < Nvz; ++k){

                int idx3 = (i * Nvy + j) * Nvz + k;
                double tmp = 0.0;
                double norm_l = std::sqrt(lx_h[i]*lx_h[i] + ly_h[j]*ly_h[j] + lz_h[k]*lz_h[k]);

                #pragma omp simd reduction(+:tmp)
                for (int r = 0; r < N_gl; ++r){
                    tmp += 16*pi*pi*b_gamma*gl_wts[r]*std::pow(gl_nodes[r], gamma+2)*sincc(pi*gl_nodes[r]*norm_l/L);
                }

                beta2_h[idx3] = tmp;

            }
        }
    }

    } // End of parallel region

    // Transfer precomputed data to the device memory
    cudaMemcpy(alpha1, alpha1_h.data(), batch_size * Nvx * Nvy * Nvz * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(beta1, beta1_h.data(), N_gl * Nvx * Nvy * Nvz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(beta2, beta2_h.data(), Nvx * Nvy * Nvz * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(gl_wts, gl_wts_h.data(), N_gl * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gl_nodes, gl_nodes_h.data(), N_gl * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(spherical_wts, spherical_quadrature->getWeights().data(), N_spherical * sizeof(double), cudaMemcpyHostToDevice);

}; // End of precomputeTransformWeights

// Compute the collision operator using FFTW
void BoltzmannOperator<CUDA_Backend>::computeCollision(double * Q, const double * f_in) {

    // Retrieve information from the quadrature objects
    int N_gl = gl_quadrature->getNumberOfPoints();
    int N_spherical = spherical_quadrature->getNumberOfPoints();

    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points
    int batch_size = N_gl * N_spherical; // Total number of grid_size batches
    double fft_scale = 1.0 / grid_size; // Scaling used to normalize transforms

    // T0-DO: Tune these parameters...
    //
    // 
    int num_blocks = 32; // Number of blocks for the CUDA kernel
    int num_threads = 256; // Number of threads per block

    // Initialize the input as a complex array
    copy_to_complex<<<num_blocks, num_threads>>>(f, f_in, grid_size);

    // Compute f_hat = fft(f)
    cufftExecC2C(plan3d, 
                reinterpret_cast<cufftComplex*>(f), 
                reinterpret_cast<cufftComplex*>(f_hat),
                CUFFT_FORWARD);

    compute_alpha_times_f_hat<<<num_blocks, num_threads>>>(alpha1_times_f_hat, 
                                    alpha2_times_f_hat, alpha1, f_hat, 
                                    N_gl, N_spherical, Nvx, Nvy, Nvz, fft_scale);

    // Compute alpha1_times_f = ifft(alpha1_times_f_hat) and 
    // alpha2_times_f = ifft(alpha2_times_f_hat)
    cufftExecC2C(plan3d_batched, 
                reinterpret_cast<cufftComplex*>(alpha1_times_f_hat), 
                reinterpret_cast<cufftComplex*>(alpha1_times_f), 
                CUFFT_INVERSE);

    cufftExecC2C(plan3d_batched,
                reinterpret_cast<cufftComplex*>(alpha2_times_f_hat), 
                reinterpret_cast<cufftComplex*>(alpha2_times_f), 
                CUFFT_INVERSE);

    // Compute transform_prod = alpha1_times_f * alpha2_times_f
    hadamard_product<<<num_blocks, num_threads>>>(transform_prod, 
                                        alpha1_times_f, 
                                        lpha2_times_f, 
                                        batch_size * grid_size);

    // Compute transform_prod_hat = fft(transform_prod)
    cufftExecC2C(plan3d_batched, 
                reinterpret_cast<cufftComplex*>(transform_prod), 
                reinterpret_cast<cufftComplex*>(transform_prod_hat),
                CUFFT_FORWARD);

    // Compute Q_gain_hat[i,j,k] = \sum_{r,s} gl_wts[r] * spherical_wts[s] *
    //                     gl_nodes[r]^(gamma+2) * beta1[r,i,j,k] * transform_prod_hat[r,s,i,j,k]
    // by contracting over slices along modes 'r' and 's'
    HANDLE_CUDA_ERROR( cudaMemset(Q_gain_hat, 0.0, grid_size * sizeof(cuDoubleComplex)) ); // Initialize to zero

    HANDLE_ERROR( cutensornetContractSlices(cutensornet_handle,
                   contraction_plan,
                   {gl_wts, spherical_wts, beta1, transform_prod_hat}, // Array of pointers to input tensors
                   Q_gain_hat, // Output tensor
                   1, // int32_t accumulateOutput = 1 means we accumulate the output
                   workDesc,
                   nullptr, // nullptr means we contract over all slices instead of specifying a sliceGroup object
                   0) ); // Use the default stream

    // Compute beta2_times_f_hat = beta2 * f_hat
    compute_beta2_times_f_hat<<<num_blocks, num_threads>>>(beta2_times_f_hat, beta2, f_hat, Nvx, Nvy, Nvz, fft_scale);

    // Note: The two transforms below can be run in different streams if needed

    // Compute Q_gain = ifft(Q_gain_hat)
    cufftExecC2C(plan3d, 
                reinterpret_cast<cufftComplex*>(Q_gain_hat), 
                reinterpret_cast<cufftComplex*>(Q_gain),
                CUFFT_INVERSE);

    // Compute beta2_times_f = ifft(beta2_times_f_hat)
    cufftExecC2C(plan3d, 
                reinterpret_cast<cufftComplex*>(beta2_times_f_hat), 
                reinterpret_cast<cufftComplex*>(beta2_times_f),
                CUFFT_INVERSE);

    // Compute Q = real(Q_gain) - real(Q_loss)
    compute_Q_total<<<num_blocks, num_threads>>>(Q, Q_gain, beta2_times_f_hat, grid_size, fft_scale);

}; // End of computeCollision


// Class destructor to clean up CUDA resources
BoltzmannOperator<CUDA_Backend>::~BoltzmannOperator() {

    // Clean up cuFFT plans
    cufftDestroy(plan3d);
    cufftDestroy(plan3d_batched);

    // Free the contraction plans and 
    cutensornetDestroyWorkspaceDescriptor(workDesc);
    cutensornetDestroyContractionAutotunePreference(autotunePref);
    cutensornetDestroyContractionPlan(contraction_plan);
    cutensornetDestroyContractionOptimizerConfig(optimizerConfig); 
    cutensornetDestroyContractionOptimizerInfo(optimizerInfo);   
    cutensornetDestroyNetworkDescriptor(descNet);
    cutensornetDestroy(cutensornet_handle);

    // Free allocated arrays for the transforms and weights
    free(alpha1);
    free(beta1);
    free(beta2);

    free(f);
    free(f_hat);

    free(alpha1_times_f);
    free(alpha1_times_f_hat);
    
    free(alpha2_times_f);
    free(alpha2_times_f_hat);
    
    free(beta2_times_f);
    free(beta2_times_f_hat);
    
    free(transform_prod);
    free(transform_prod_hat);
    
    free(Q_gain_hat);
    free(Q_gain);

}; // End of destructor
