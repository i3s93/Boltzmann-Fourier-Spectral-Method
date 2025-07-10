// Demonstration of the Boltzmann collision operator for Maxwell molecules using the BKW solution to test 
// the accuracy of the solver.
//
// This specific code uses OpenMP + CUDA for parallelization.

#include <iostream>    
#include <iomanip>     
#include <string>      
#include <vector>      
#include <cmath>       
#include <algorithm>   

#include <tclap/CmdLine.h>  
#include <omp.h>
#include <cuda_runtime.h>

#include "Utilities/constants.hpp"       
#include "Utilities/statistics.hpp"      
#include "Quadratures/SphericalDesign.hpp" 
#include "Quadratures/GaussLegendre.hpp"   
#include "Collisions/CUDABoltzmannOperator.hpp"

int main(int argc, char** argv) {

    int Nv, Ns, trials;

    try {
        // Create each of the arguments
        TCLAP::CmdLine cmd("Command description message", ' ', "1.0");
        TCLAP::ValueArg<int> Nv_Arg("", "Nv", "Number of points per dimension in velocity", false, 32, "int");
        TCLAP::ValueArg<int> Ns_Arg("", "Ns", "Number of points on the unit sphere", false, 12, "int");
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

    // Test for Maxwell molecules
    const double gamma = 0;
    const double b_gamma = 1/(4*pi);

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

    // Setup the BKW solution then find the corresponding collision operator Q
    const double t = 6.5;
    const double K = 1 - std::exp(-t/6);
    const double dK = std::exp(-t/6)/6;

    // Host arrays for the BKW solution and the corresponding Q
    std::vector<double> f_bkw_h(Nv*Nv*Nv);
    std::vector<double> Q_bkw_h(Nv*Nv*Nv);

    #pragma omp parallel for collapse (2)
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            #pragma omp simd
            for (int k = 0; k < Nv; ++k){
                int idx3 = (i * Nv + j) * Nv + k;
                double r_sq = vx[i]*vx[i] + vy[j]*vy[j] + vz[k]*vz[k];;
                
                // Compute the BKW solution
                f_bkw_h[idx3] = std::exp(-(r_sq)/(2*K))*((5*K-3)/K+(1-K)/(std::pow(K,2))*(r_sq));
                f_bkw_h[idx3] *= 1/(2*std::pow(2*pi*K, 1.5));

                // Compute the derivative of f
                Q_bkw_h[idx3] = (-3/(2*K) + r_sq/(2*std::pow(K,2)))*f_bkw[idx3];
                Q_bkw_h[idx3] += 1/(2*std::pow(2*pi*K, 1.5))*std::exp(-r_sq/(2*K))*(3/(std::pow(K,2)) + (K-2)/(std::pow(K,3))*r_sq);
                Q_bkw_h[idx3] *= dK;
            }
        }
    }

    // Compute the quadrature rules and store their information in the solver
    auto gl_quadrature = std::make_shared<GaussLegendreQuadrature>(Nv, 0, R);
    auto spherical_quadrature = std::make_shared<SphericalDesign>(Ns);
    
    // Allocate space for the output of the collision operator on the host
    // We perform the evaluation on the device and then copy the results back to the host
    // for error measurements
    std::vector<double> = Q_h(Nv*Nv*Nv);

    // Next, we need to allocate the device arrays for the evaluation
    double * f_bkw;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&f_bkw, Nv*Nv*Nv*sizeof(double)));

    double * Q;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&Q, Nv*Nv*Nv*sizeof(double)));

    // Copy the BKW solution to the device
    HANDLE_CUDA_ERROR( cudaMemcpy(f_bkw, f_bkw_h.data(), Nv*Nv*Nv*sizeof(double), cudaMemcpyHostToDevice));

    // Create the Boltzmann operator for Maxwell molecules using the CUDA backend
    BoltzmannOperator<CUDA_Backend> collision_operator(gl_quadrature, spherical_quadrature, 
        Nv, Nv, Nv, gamma, b_gamma, L);

    // Initialize the operator and precompute the transform weights
    double initialize_time = omp_get_wtime();
    collision_operator.initialize();
    double initialize_end_time = omp_get_wtime();
    double initialize_total_time = initialize_end_time - initialize_time;
    std::cout << "Initialization time (s): " << initialize_total_time << " seconds\n";

    double precompute_time = omp_get_wtime();
    collision_operator.precomputeTransformWeights();
    double precompute_end_time = omp_get_wtime();
    double precompute_total_time = precompute_end_time - precompute_time;
    std::cout << "Precomputation time (s): " << precompute_total_time << " seconds\n";

    // Container to hold the timing data
    std::vector<double> collision_times;
    collision_times.reserve(trials);

    // Time the collision operator computation and store the results for statistics
    for(int trial_idx = 0; trial_idx < trials; ++trial_idx){

        double start_time = omp_get_wtime();
        collision_operator(Q, f_bkw); 
        double end_time = omp_get_wtime();
        collision_times.push_back(end_time - start_time);

    }

    print_stats_summary("CUDA", collision_times);

    // Next we need to copy the results back to the host
    HANDLE_CUDA_ERROR( cudaMemcpy(Q_h.data(), Q, Nv*Nv*Nv*sizeof(double), cudaMemcpyDeviceToHost));

    // Check the errors in the different norms and print them to the console
    double err_L1 = 0;
    double err_L2 = 0;
    double err_Linf = 0;
    double abs_diff;

    #pragma omp parallel for simd reduction(+:err_L1, err_L2, err_Linf)
    for (int idx = 0; idx < Nv*Nv*Nv; ++idx){
        abs_diff = std::abs(Q_h[idx] - Q_bkw_h[idx]);
        err_L1 += abs_diff;
        err_L2 += std::pow(abs_diff,2);
        err_Linf = std::max(err_Linf, abs_diff);
    }

    // L1 and L2 errors need to be further modified
    err_L1 *= std::pow(dv,3);
    err_L2 *= std::pow(dv,3);
    err_L2 = std::sqrt(err_L2);

    std::cout << "Approximation errors:\n";
    std::cout << "L1 error: " << err_L1 << "\n";
    std::cout << "L2 error: " << err_L2 << "\n";
    std::cout << "Linf error: " << err_Linf << "\n\n";

    // Free the allocated memory on the device
    HANDLE_CUDA_ERROR( cudaFree(f_bkw) );
    HANDLE_CUDA_ERROR( cudaFree(Q) );
  
    return 0;
}
