// A benchmark for batched 3D transforms using FFTW
//
// This specific code uses OpenMP for parallelization.

#include <iostream>    // For input/output operations
#include <iomanip>     // For formatting output
#include <string>      // For string manipulation
#include <complex>     // For std::complex
#include <vector>      // For std::vector
#include <cmath>       // For mathematical functions like std::sqrt, std::pow, std::exp
#include <algorithm>   // For std::max

#include <tclap/CmdLine.h>  // For command-line argument parsing
#include <omp.h>           // For OpenMP parallelization
#include <fftw3.h>        // For FFTW library functions

#include "Utilities/constants.hpp"       // For constants like pi
#include "Utilities/statistics.hpp"      // For statistical functions

int main(int argc, char** argv) {

    int Nv, batch_size, trials;

    try {
        // Create each of the arguments
        TCLAP::CmdLine cmd("Command description message", ' ', "1.0");
        TCLAP::ValueArg<int> Nv_Arg("", "Nv", "Number of points per dimension in velocity", false, 32, "int");
        TCLAP::ValueArg<int> batch_Arg("", "batch_size", "Batch size for the FFTs", false, 256, "int");
        TCLAP::ValueArg<int> t_Arg("t", "trials", "Number of trials to use for statistics", false, 1, "int");

        cmd.add(Nv_Arg);
        cmd.add(batch_Arg);
        cmd.add(t_Arg);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Assign parsed values to variables
        Nv = Nv_Arg.getValue();
        batch_size = batch_Arg.getValue();
        trials = t_Arg.getValue();

        std::cout << "\nRun arguments:" << "\n";
        std::cout << "Nv = " << Nv << "\n";
        std::cout << "Batch size = " << batch_size << "\n";
        std::cout << "trials = " << trials << "\n";
    } catch (TCLAP::ArgException &e)
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << "\n"; }

    // Support constants and mesh information for the spectral method
    const double S = 5;
    const double R = 2*S;
    const double L = ((3 + std::sqrt(2))/2)*S;

    // Build the velocity domain as a tensor product grid
    const double dv = 2*L/Nv;
    std::vector<double> vx(Nv);

    for (int i = 0; i < Nv; ++i){
        vx[i] = -L + dv/2 + i*dv;
    }

    std::vector<double> vy = vx;
    std::vector<double> vz = vx;

    // Setup a simple distribution function, e.g., the BKW solution
    const double t = 6.5;
    const double K = 1 - std::exp(-t/6);
    const int grid_size = Nv * Nv * Nv;
    double scale_factor = 1.0 / grid_size;
    std::complex<double>* f_bkw = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));
    
    #pragma omp parallel for collapse (3)
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            for (int k = 0; k < Nv; ++k){
                int idx3 = (i * Nv + j) * Nv + k;
                double r_sq = vx[i]*vx[i] + vy[j]*vy[j] + vz[k]*vz[k];
                f_bkw[idx3] = std::exp(-(r_sq)/(2*K))*((5*K-3)/K+(1-K)/(std::pow(K,2))*(r_sq));
                f_bkw[idx3] *= 1/(2*std::pow(2*pi*K, 1.5));
            }
        }
    }

    // Allocate storage for the batching experiment
    std::complex<double>* f     = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    std::complex<double>* f_hat = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    
    // Fill the batch with the same distribution function
    #pragma omp parallel for collapse (4)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                for (int k = 0; k < Nv; ++k){
                    int idx3 = (i * Nv + j) * Nv + k;
                    int idx4 = b * grid_size + (i * Nv + j) * Nv + k;
                    f[idx4] = f_bkw[idx3];
                }
            }
        }
    }

    // Approach 1: Use FFTW with the plan many interface for batched 3D transforms
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());    

    int batched_rank = 3; // Each FFT is applied to a three-dimensional row-major array
    int batched_dims[] = {Nv, Nv, Nv}; // Dimensions of the arrays used in each transform
    int idist = grid_size; // Input array is separated by idist elements
    int odist = idist; // Output array is separated by odist elements
    int istride = 1; // Input array is contiguous in memory
    int ostride = 1; // Output array is contiguous in memory
    int *inembed = batched_dims; // The array is not embedded in a larger array
    int *onembed = batched_dims; // The array is not embedded in a larger array

    fftw_plan fft_batch_plan_many = fftw_plan_many_dft(batched_rank, batched_dims, batch_size,
                                    reinterpret_cast<fftw_complex*>(f), 
                                    inembed, istride, idist,
                                    reinterpret_cast<fftw_complex*>(f_hat),
                                    onembed, ostride, odist,
                                    FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_plan ifft_batch_plan_many = fftw_plan_many_dft(batched_rank, batched_dims, batch_size,
                                    reinterpret_cast<fftw_complex*>(f_hat), 
                                    inembed, istride, idist,
                                    reinterpret_cast<fftw_complex*>(f),
                                    onembed, ostride, odist,
                                    FFTW_BACKWARD, FFTW_ESTIMATE);

    std::vector<double> times_plan_many;
    times_plan_many.reserve(trials);

    for(int trial_idx = 0; trial_idx < trials; ++trial_idx){
        double start_time = omp_get_wtime();
        fftw_execute(fft_batch_plan_many);
        double end_time = omp_get_wtime();
        times_plan_many.push_back(end_time - start_time);
    }

    #pragma omp parallel for collapse (4)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                for (int k = 0; k < Nv; ++k){
                    int idx4 = b * grid_size + (i * Nv + j) * Nv + k;
                    f_hat[idx4] *= scale_factor;
                }
            }
        }
    }

    // Take the inverse transform and measure the error
    fftw_execute(ifft_batch_plan_many);

    double L1_error_plan_many = 0.0;

    #pragma omp parallel for collapse (4) reduction(+:L1_error_plan_many)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                for (int k = 0; k < Nv; ++k){
                    int idx3 = (i * Nv + j) * Nv + k;
                    int idx4 = b * grid_size + (i * Nv + j) * Nv + k;
                    L1_error_plan_many += std::abs(f_bkw[idx3] - f[idx4].real());
                }
            }
        }
    }

    L1_error_plan_many *= dv * dv * dv;
    std::cout << "Approximation error (Plan many):\n";
    std::cout << "L1 error: " << L1_error_plan_many << "\n";
    print_stats_summary("Plan many", times_plan_many);


    // Approach 2: Use FFTW with the new array execute interface
    // We use dummy 3D arrays to create plans for the forward and inverse transforms 
    std::complex<double>* data     = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));
    std::complex<double>* data_hat = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));

    // Each FFT/iFFT will be single threaded, but will parallelize over batches
    fftw_plan_with_nthreads(1);

    fftw_plan fft_3d_plan  = fftw_plan_dft_3d(Nv, Nv, Nv,
                                          reinterpret_cast<fftw_complex*>(data),
                                          reinterpret_cast<fftw_complex*>(data_hat),
                                          FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_plan ifft_3d_plan = fftw_plan_dft_3d(Nv, Nv, Nv,
                                           reinterpret_cast<fftw_complex*>(data_hat),
                                           reinterpret_cast<fftw_complex*>(data),
                                           FFTW_BACKWARD, FFTW_ESTIMATE);


     std::vector<double> times_manual_batch;
     times_manual_batch.reserve(trials);

     // Note: We should ignore the time to startup threads
     for(int trial_idx = 0; trial_idx < trials; ++trial_idx){

        #pragma omp parallel
        {
            double start_time, end_time;

            #pragma omp master
            start_time = omp_get_wtime();

            #pragma omp for
            for(int b = 0; b < batch_size; ++b){
                fftw_execute_dft(fft_3d_plan, 
                                reinterpret_cast<fftw_complex*>(f + b * grid_size), 
                                reinterpret_cast<fftw_complex*>(f_hat + b * grid_size));
            }

            #pragma omp master
            {
                end_time = omp_get_wtime();
                times_manual_batch.push_back(end_time - start_time);
            }
        }
     }
 
     #pragma omp parallel for collapse (4)
     for (int b = 0; b < batch_size; ++b) {
         for (int i = 0; i < Nv; ++i){
             for (int j = 0; j < Nv; ++j){
                 for (int k = 0; k < Nv; ++k){
                     int idx4 = b * grid_size + (i * Nv + j) * Nv + k;
                     f_hat[idx4] *= scale_factor;
                 }
             }
         }
     }
 
     // Take the inverse transform and measure the error
    #pragma omp parallel for
    for(int b = 0; b < batch_size; ++b){
        fftw_execute_dft(ifft_3d_plan, 
                        reinterpret_cast<fftw_complex*>(f_hat + b * grid_size), 
                        reinterpret_cast<fftw_complex*>(f + b * grid_size));
    }

    double L1_error_manual_batch = 0.0;
 
     #pragma omp parallel for collapse (4) reduction(+:L1_error_manual_batch)
     for (int b = 0; b < batch_size; ++b) {
         for (int i = 0; i < Nv; ++i){
             for (int j = 0; j < Nv; ++j){
                 for (int k = 0; k < Nv; ++k){
                     int idx3 = (i * Nv + j) * Nv + k;
                     int idx4 = b * grid_size + (i * Nv + j) * Nv + k;
                     L1_error_manual_batch += std::abs(f_bkw[idx3] - f[idx4].real());
                 }
             }
         }
     }
 
     L1_error_manual_batch *= dv * dv * dv;
     std::cout << "Approximation error (manual batching):\n";
     std::cout << "L1 error: " << L1_error_manual_batch << "\n";
     print_stats_summary("Manual batching", times_manual_batch);

    // Free any allocated memory for arrays and plans
    fftw_free(f_bkw);
    
    fftw_free(f);
    fftw_free(f_hat);
    
    fftw_free(data);
    fftw_free(data_hat);

    fftw_destroy_plan(fft_batch_plan_many);
    fftw_destroy_plan(ifft_batch_plan_many);
    
    fftw_destroy_plan(fft_3d_plan);
    fftw_destroy_plan(ifft_3d_plan);
  
    return 0;
}
