// A benchmark for batched 3D transforms using cuFFT
//
// This specific code uses CUDA for parallelization.

#include <iostream>    
#include <iomanip>    
#include <string>     
#include <vector>     
#include <cmath>      
#include <algorithm>  

#include <tclap/CmdLine.h>  
#include <omp.h> 
#include <cuda_runtime.h>
#include <cuComplex.h> 
#include <cufft.h>     

#include "Utilities/constants.hpp"   
#include "Utilities/statistics.hpp" 

#define HANDLE_CUDA_ERROR(x)                                      \
{                                                                 \
    const auto err = x;                                           \
    if (err != cudaSuccess) {                                     \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)    \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE);                                  \
    }                                                             \
}

#define CUFFT_CALL(x)                                                     \
{                                                                         \
    const auto status = x;                                                \
    if (status != CUFFT_SUCCESS) {                                        \
        std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__     \
                    << " code " << status << " (" << cufftGetErrorString(status) << ")" << std::endl; \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
}

// Function that maps an error code from cufft to a string for printing
std::string cufftGetErrorString(cufftResult error);

// CUDA kernel for scaling the data
__global__ void scaling_kernel(cuDoubleComplex * data, const double scale, const int N);

int main(int argc, char** argv) {

    int Nv, Ns, trials;

    try {
        // Create each of the arguments
        TCLAP::CmdLine cmd("Command description message", ' ', "1.0");
        TCLAP::ValueArg<int> Nv_Arg("", "Nv", "Number of points per dimension in velocity", false, 32, "int");
        TCLAP::ValueArg<int> Ns_Arg("", "Ns", "Number of spherical quadrature points", false, 32, "int");
        TCLAP::ValueArg<int> t_Arg("t", "trials", "Number of trials to use for statistics", false, 1, "int");

        cmd.add(Nv_Arg);
        cmd.add(Ns_Arg);
        cmd.add(t_Arg);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Assign parsed values to variables
        Nv = Nv_Arg.getValue();
        Ns = Ns_Arg.getValue();
        trials = t_Arg.getValue();

        std::cout << "\nRun arguments:" << "\n";
        std::cout << "Nv = " << Nv << "\n";
        std::cout << "Ns = " << Ns << "\n";
        std::cout << "trials = " << trials << "\n";
    } catch (TCLAP::ArgException &e)
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << "\n"; }

    // Support constants and mesh information for the spectral method
    const double S = 5;
    const double L = ((3 + std::sqrt(2))/2)*S;

    // Build the velocity domain as a tensor product grid
    const double dv = 2*L/Nv;
    std::vector<double> vx(Nv);

    for (int i = 0; i < Nv; ++i){
        vx[i] = -L + dv/2 + i*dv;
    }

    std::vector<double> vy = vx;
    std::vector<double> vz = vx;

    // Setup a simple input on the CPU
    const int grid_size = Nv * Nv * Nv;
    const int batch_size = Ns * Nv;
    double scale_factor = 1.0 / grid_size;
    std::vector<cuDoubleComplex> f_h(grid_size);   
 
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            for (int k = 0; k < Nv; ++k){
                int idx3 = (i * Nv + j) * Nv + k;
                f_h[idx3].x = 1;
                f_h[idx3].y = 0;
            }
        }
    }

    // Allocate storage for the batching experiment on the device
    cuDoubleComplex * f;
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&f, batch_size * grid_size * sizeof(cuDoubleComplex)) );

    cuDoubleComplex * f_hat;
    HANDLE_CUDA_ERROR( cudaMalloc((void **)&f_hat, batch_size * grid_size * sizeof(cuDoubleComplex)) );

    // Let's also make some cuda streams to overlap some of the copies
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);

    for(int i = 0; i < num_streams; ++i){HANDLE_CUDA_ERROR( cudaStreamCreate(&streams[i]) );}

    // Fill the batch with the same distribution function
    for (int b = 0; b < batch_size; ++b) {
        int sid = b % num_streams; 
        HANDLE_CUDA_ERROR( cudaMemcpyAsync(f + b * grid_size, f_h.data(), grid_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, streams[sid]) ); 
    }

    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

    // Use cuFFT with the plan many interface for batched 3D transforms
    cufftHandle plan3d_batched;

    int batched_rank = 3; // Each FFT is applied to a three-dimensional row-major array
    std::vector<int> batched_dims{Nv, Nv, Nv}; // Dimensions of the arrays used in each transform
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

    std::vector<double> times_plan_many;
    times_plan_many.reserve(trials);

    for(int trial_idx = 0; trial_idx < trials; ++trial_idx){
        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

        double start_time = omp_get_wtime();
        
        CUFFT_CALL( cufftExecZ2Z(plan3d_batched,
                    reinterpret_cast<cufftDoubleComplex*>(f), 
                    reinterpret_cast<cufftDoubleComplex*>(f_hat), 
                    CUFFT_FORWARD) );

        HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
        
        double end_time = omp_get_wtime();
        times_plan_many.push_back(end_time - start_time);
    }

    // Launch the scaling kernel to normalize the output of the transform
    int num_threads_per_block = 256;
    int num_blocks = std::max( int( grid_size / (4 * num_threads_per_block) ), 1 );
    scaling_kernel<<<num_blocks, num_threads_per_block>>>(f_hat, scale_factor, batch_size * grid_size);
    HANDLE_CUDA_ERROR( cudaGetLastError() );
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

    // Take the inverse transform 
    CUFFT_CALL( cufftExecZ2Z(plan3d_batched,
                reinterpret_cast<cufftDoubleComplex*>(f_hat), 
                reinterpret_cast<cufftDoubleComplex*>(f), 
                CUFFT_INVERSE) );

    // Copy the result from the device back to the host
    std::vector<cuDoubleComplex> result(batch_size * grid_size);

    for (int b = 0; b < batch_size; ++b) {
        int sid = b % num_streams; 
        HANDLE_CUDA_ERROR( cudaMemcpyAsync(result.data() + b * grid_size, f + b * grid_size, grid_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, streams[sid]) ); 
    }

    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

    double L1_error_plan_many = 0.0;

    #pragma omp parallel for simd collapse (4) reduction(+:L1_error_plan_many)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                for (int k = 0; k < Nv; ++k){
                    int idx3 = (i * Nv + j) * Nv + k;
                    int idx4 = b * grid_size + (i * Nv + j) * Nv + k;
                    L1_error_plan_many += std::abs(f_h[idx3].x - result[idx4].x);
                }
            }
        }
    }

    L1_error_plan_many *= dv * dv * dv;
    std::cout << "Approximation error (Plan many):\n";
    std::cout << "L1 error: " << L1_error_plan_many << "\n";
    print_stats_summary("Plan many", times_plan_many);

    // Free any allocated memory for arrays and plans
    CUFFT_CALL( cufftDestroy(plan3d_batched) );

    HANDLE_CUDA_ERROR( cudaFree(f) );
    HANDLE_CUDA_ERROR( cudaFree(f_hat) );
  
    // Release the cuda streams
    for(int i = 0; i < num_streams; ++i) {HANDLE_CUDA_ERROR( cudaStreamDestroy(streams[i]) );}

    return 0;
}

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

__global__ void scaling_kernel(cuDoubleComplex * data, const double scale, const int N) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int idx = tid; idx < N; idx += num_threads){
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

