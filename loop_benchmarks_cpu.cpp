// A benchmark for performance critical nested loops.
//
// This specific code uses OpenMP for parallelization.

#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <cmath>

#include <tclap/CmdLine.h>  
#include <omp.h>           
#include <fftw3.h>          

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

    // Setup the arrays that will be used in the loops
    const int grid_size = Nv * Nv * Nv;
    const int batch_size = Ns * Nv;

    std::complex<double>* alpha1 = (std::complex<double>*) fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    double* beta1 = (double*) fftw_malloc(Nv * grid_size * sizeof(double));
    double* beta2 = (double*) fftw_malloc(grid_size * sizeof(double));

    std::complex<double>* f_hat = (std::complex<double>*) fftw_malloc(grid_size * sizeof(std::complex<double>));
    std::complex<double>* alpha1_times_f_hat = (std::complex<double>*) fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    std::complex<double>* alpha2_times_f_hat = (std::complex<double>*) fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    std::complex<double>* transform_prod_hat = (std::complex<double>*) fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    std::complex<double>* Q_gain_hat = (std::complex<double>*) fftw_malloc(grid_size * sizeof(std::complex<double>));

    std::vector<double> gl_nodes(Nv);
    std::vector<double> gl_wts(Nv);
    std::vector<double> spherical_wts(Ns);
    
    const double gamma = 0.0;
    const double b_gamma = 1/(4*pi);
    const double fft_scale = 1 / grid_size;
    double start_time = 0.0;
    double end_time = 0.0;
    double total_time = 0.0;

    // Rather than store each timing instance, we will report the total time over all trials
    // since the goal is to minimize the total runtime of the nested loops.

    // Fill the arrays with some values
    #pragma omp parallel
    {

    #pragma omp for
    for (int idx = 0; idx < batch_size * grid_size; ++idx) {
        alpha1[idx] = std::complex<double>(0.75*idx, 0.25*idx);
        alpha1_times_f_hat[idx] = std::complex<double>(0.5*idx, 0.5*idx);
        alpha2_times_f_hat[idx] = std::complex<double>(0.25*idx, 0.75*idx);
        transform_prod_hat[idx] = std::complex<double>(0.1*idx, 0.9*idx);
    }

    #pragma omp for
    for (int idx = 0; idx < Nv * grid_size; ++idx) {
        beta1[idx] = 0.5*idx;
    }

    #pragma omp for
    for (int idx = 0; idx < grid_size; ++idx) {
        beta2[idx] = 0.75*idx;
        f_hat[idx] = std::complex<double>(0.5*idx, 0.5*idx);
        Q_gain_hat[idx] = 0.0;
    }

    #pragma omp for
    for (int idx = 0; idx < Nv; ++idx) {
        gl_nodes[idx] = 0.1 * idx; // Example values
        gl_wts[idx] = 0.2 * idx;   // Example values
    }

    #pragma omp for
    for (int idx = 0; idx < Ns; ++idx) {
        spherical_wts[idx] = 1.0 / Ns; // If these are constant the compiler can optimize them out
    }

    } // End of parallel region


    std::cout << "\nInitialization complete...\n";

    // Pattern 1: Nested loops with OpenMP collapse and SIMD

    std::cout << "\nBeginning the experiments for pattern 1...\n";
    std::cout << "Method 1: omp for collapse(5) simd...\n";
    total_time = 0.0;

    for (int trial = 0; trial < trials; ++trial) {

        #pragma omp parallel
        {

        // Start the time measurement
        #pragma omp master
        {
            start_time = omp_get_wtime();
        }

        #pragma omp for collapse(5) simd
        for (int r = 0; r < Nv; ++r){
            for (int s = 0; s < Ns; ++s){
                for (int i = 0; i < Nv; ++i){
                    for (int j = 0; j < Nv; ++j){
                        for (int k = 0; k < Nv; ++k){
                            int idx3 = (i * Nv + j) * Nv + k;
                            int idx5 = ((((r) * Ns + s) * Nv + i) * Nv + j) * Nv + k;
                            alpha1_times_f_hat[idx5] = fft_scale*alpha1[idx5]*f_hat[idx3];
                            alpha2_times_f_hat[idx5] = fft_scale*std::conj(alpha1[idx5])*f_hat[idx3];
                        }
                    }
                }
            }
        }

        // End the time measurement
        #pragma omp master
        {
            end_time = omp_get_wtime();
            total_time += (end_time - start_time);
        }

        } // End of parallel region

    } // End of trials

    std::cout << "\nTotal time (s): " << total_time << "\n";

 
    std::cout << "Method 2: omp for collapse(4) with inner omp simd...\n";
    total_time = 0.0;

    for (int trial = 0; trial < trials; ++trial) {

        #pragma omp parallel
        {

        // Start the time measurement
        #pragma omp master
        {
            start_time = omp_get_wtime();
        }

        #pragma omp for collapse(4) 
        for (int r = 0; r < Nv; ++r){
            for (int s = 0; s < Ns; ++s){
                for (int i = 0; i < Nv; ++i){
                    for (int j = 0; j < Nv; ++j){
                        #pragma omp simd
                        for (int k = 0; k < Nv; ++k){
                            int idx3 = (i * Nv + j) * Nv + k;
                            int idx5 = ((((r) * Ns + s) * Nv + i) * Nv + j) * Nv + k;
                            alpha1_times_f_hat[idx5] = fft_scale*alpha1[idx5]*f_hat[idx3];
                            alpha2_times_f_hat[idx5] = fft_scale*std::conj(alpha1[idx5])*f_hat[idx3];
                        }
                    }
                }
            }
        }

        // End the time measurement
        #pragma omp master
        {
            end_time = omp_get_wtime();
            total_time += (end_time - start_time);
        }

        } // End of parallel region

    } // End of trials

    std::cout << "\nTotal time (s): " << total_time << "\n";


    // Pattern 2: Nested loops with reduction

    std::cout << "\nBeginning the experiments for pattern 2...\n";
    std::cout << "Method 1: omp for collapse(5) reduction(+:Q_gain_hat[:grid_size])...\n";
    total_time = 0.0;

    for (int trial = 0; trial < trials; ++trial) {

        #pragma omp parallel
        {

        // Start the time measurement
        #pragma omp master
        {
            start_time = omp_get_wtime();
        }

        #pragma omp for collapse(5) reduction(+:Q_gain_hat[:grid_size])
        for (int r = 0; r < Nv; ++r){
            for (int s = 0; s < Ns; ++s){
                for (int i = 0; i < Nv; ++i){
                    for (int j = 0; j < Nv; ++j){
                        for (int k = 0; k < Nv; ++k){
                            int idx3 = (i * Nv + j) * Nv + k;
                            int idx4 = (((r * Nv + i) * Nv + j) * Nv + k);
                            int idx5 = ((((r) * Ns + s) * Nv + i) * Nv + j) * Nv + k;
                            Q_gain_hat[idx3] += fft_scale*gl_wts[r]*spherical_wts[s]*std::pow(gl_nodes[r], gamma+2)*beta1[idx4]*transform_prod_hat[idx5];
                        }
                    }
                }
            }
        }

        // End the time measurement
        #pragma omp master
        {
            end_time = omp_get_wtime();
            total_time += (end_time - start_time);
        }

        } // End of parallel region

    } // End of trials

    std::cout << "\nTotal time (s): " << total_time << "\n";

    std::cout << "Method 2: omp for collapse(2) reduction(+:Q_gain_hat[:grid_size])...\n";
    total_time = 0.0;

    for (int trial = 0; trial < trials; ++trial) {

        #pragma omp parallel
        {

        // Start the time measurement
        #pragma omp master
        {
            start_time = omp_get_wtime();
        }

        #pragma omp for collapse(2) reduction(+:Q_gain_hat[:grid_size])
        for (int r = 0; r < Nv; ++r){
            for (int s = 0; s < Ns; ++s){

                double weight = fft_scale * gl_wts[r] * spherical_wts[s] * std::pow(gl_nodes[r], gamma + 2);

                #pragma omp collapse(3) simd
                for (int i = 0; i < Nv; ++i){
                    for (int j = 0; j < Nv; ++j){
                        for (int k = 0; k < Nv; ++k){
                            int idx3 = (i * Nv + j) * Nv + k;
                            int idx4 = (((r * Nv + i) * Nv + j) * Nv + k);
                            int idx5 = ((((r) * Ns + s) * Nv + i) * Nv + j) * Nv + k;
                            Q_gain_hat[idx3] += weight*beta1[idx4]*transform_prod_hat[idx5];
                        }
                    }
                }
            }
        }

        // End the time measurement
        #pragma omp master
        {
            end_time = omp_get_wtime();
            total_time += (end_time - start_time);
        }

        } // End of parallel region

    } // End of trials

    std::cout << "\nTotal time (s): " << total_time << "\n";

    // Free allocated memory
    fftw_free(alpha1);
    fftw_free(beta1);
    fftw_free(beta2);
    fftw_free(f_hat);
    fftw_free(alpha1_times_f_hat);
    fftw_free(alpha2_times_f_hat);
    fftw_free(transform_prod_hat);
    fftw_free(Q_gain_hat);

    return 0;
}

