#include "FFTWBoltzmannOperator.hpp"

// Custom reduction for complex types since OpenMP does not provide this
/* This uses a custom reducer... not supported by NVIDIA OpenMP compilers
#pragma omp declare	\
reduction(	\
	+ : \
	std::complex<double> :	\
	omp_out += omp_in )	\
initializer( omp_priv = omp_orig )
*/

// Initialize plans and arrays for the FFTW backend
void BoltzmannOperator<FFTW_Backend>::initialize() {

    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points
    int N_gl = gl_quadrature->getNumberOfPoints(); // Number of Gauss-Legendre quadrature points
    int N_spherical = spherical_quadrature->getNumberOfPoints(); // Number of spherical quadrature points
    int batch_size = N_gl * N_spherical; // Total number of grid_size batches

    // Allocations for the arrays used the operator
    // Note: alpha2 = conj(alpha1) so we don't store it
    alpha1 = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>)); 
    beta1 = (double*)fftw_malloc(N_gl * grid_size * sizeof(double));
    beta2 = (double*)fftw_malloc(grid_size * sizeof(double));

    data = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));
    data_hat = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));

    f = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));
    f_hat = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));
        
    alpha1_times_f = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    alpha1_times_f_hat = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));

    alpha2_times_f = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    alpha2_times_f_hat = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    
    transform_prod = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));
    transform_prod_hat = (std::complex<double>*)fftw_malloc(batch_size * grid_size * sizeof(std::complex<double>));

    Q_gain = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));
    Q_gain_hat = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));

    beta2_times_f = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));
    beta2_times_f_hat = (std::complex<double>*)fftw_malloc(grid_size * sizeof(std::complex<double>));

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

    // Check if there is a wisdom file available
    if (fftw_import_wisdom_from_filename(wisdom_fname.c_str()) == 0) {
        std::cout << "Failed to import wisdom from file: " << wisdom_fname << "\n";
    }

    fft_plan  = fftw_plan_dft_3d(Nvx, Nvy, Nvz,
        reinterpret_cast<fftw_complex*>(data),
        reinterpret_cast<fftw_complex*>(data_hat),
        FFTW_FORWARD, FFTW_EXHAUSTIVE);

    ifft_plan = fftw_plan_dft_3d(Nvx, Nvy, Nvz,
            reinterpret_cast<fftw_complex*>(data_hat),
            reinterpret_cast<fftw_complex*>(data),
            FFTW_BACKWARD, FFTW_EXHAUSTIVE);

    // Export wisdom immediately after plan creation using the specified filename
    fftw_export_wisdom_to_filename(wisdom_fname.c_str());

}; // End of initialize

// Precompute the transform weights alpha1, beta1, beta2
void BoltzmannOperator<FFTW_Backend>::precomputeTransformWeights() {

    int N_gl = gl_quadrature->getNumberOfPoints(); // Number of Gauss-Legendre quadrature points
    int N_spherical = spherical_quadrature->getNumberOfPoints(); // Number of spherical quadrature points
    //int batch_size = N_gl * N_spherical; // Total number of grid_size batches

    // Extract the quadrature weights and nodes (using shallow copies)
    const std::vector<double>& gl_wts = gl_quadrature->getWeights();
    const std::vector<double>& gl_nodes = gl_quadrature->getNodes();
    const std::vector<double>& sx = spherical_quadrature->getx();
    const std::vector<double>& sy = spherical_quadrature->gety();
    const std::vector<double>& sz = spherical_quadrature->getz();

    if (lx.empty() || ly.empty() || lz.empty()) {
        throw std::runtime_error("lx, ly, or lz is not initialized");
    }

    if (sx.empty() || sy.empty() || sz.empty()) {
        throw std::runtime_error("sx, sy, or sz is not initialized");
    }

    #pragma omp parallel
    {

    // Compute the complex transform weights alpha1
    // Note that we do not compute alpha2 since it is the conjugate of alpha1
    #pragma omp for collapse(4)
    for (int r = 0; r < N_gl; ++r){
        for (int s = 0; s < N_spherical; ++s){
            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nvz; ++k){
                        int idx5 = ((((r) * N_spherical + s) * Nvx + i) * Nvy + j) * Nvz + k;
                        double l_dot_sigma = lx[i]*sx[s] + ly[j]*sy[s] + lz[k]*sz[s];
                        alpha1[idx5] = std::exp(std::complex<double>(0,-(pi/(2*L))*gl_nodes[r]*l_dot_sigma));                   
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
                    double norm_l = std::sqrt(lx[i]*lx[i] + ly[j]*ly[j] + lz[k]*lz[k]);
                    beta1[idx4] = 4*pi*b_gamma*sincc(pi*gl_nodes[r]*norm_l/(2*L));
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
                double norm_l = std::sqrt(lx[i]*lx[i] + ly[j]*ly[j] + lz[k]*lz[k]);

                #pragma omp simd reduction(+:tmp)
                for (int r = 0; r < N_gl; ++r){
                    tmp += 16*pi*pi*b_gamma*gl_wts[r]*std::pow(gl_nodes[r], gamma+2)*sincc(pi*gl_nodes[r]*norm_l/L);
                }

                beta2[idx3] = tmp;

            }
        }
    }

    } // End of parallel region

}; // End of precomputeTransformWeights

// Compute the collision operator using FFTW
void BoltzmannOperator<FFTW_Backend>::computeCollision(double * Q, const double * f_in) {

    // Retrieve information from the quadrature objects
    int N_gl = gl_quadrature->getNumberOfPoints();
    int N_spherical = spherical_quadrature->getNumberOfPoints();    

    const std::vector<double>& gl_wts = gl_quadrature->getWeights();
    const std::vector<double>& gl_nodes = gl_quadrature->getNodes();
    const std::vector<double>& spherical_wts = spherical_quadrature->getWeights();

    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points
    int batch_size = N_gl * N_spherical; // Total number of grid_size batches
    double fft_scale = 1.0 / grid_size; // Scaling used to normalize transforms

    #pragma omp parallel
    {

    // Initialize the input as a complex array
    #pragma omp for collapse(2)
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            #pragma omp simd
            for (int k = 0; k < Nvz; ++k){
                int idx3 = (i * Nvy + j) * Nvz + k;
                f[idx3] = f_in[idx3];
                Q_gain_hat[idx3] = 0;
            }
        }
    }

    #pragma omp barrier

    // Compute f_hat = fft(f)
    #pragma omp single
    fftw_execute_dft(fft_plan, 
                    reinterpret_cast<fftw_complex*>(f), 
                    reinterpret_cast<fftw_complex*>(f_hat));

    #pragma omp barrier

    // Compute alpha1_times_f_hat and alpha2_times_f_hat
    #pragma omp for collapse(4)
    for (int r = 0; r < N_gl; ++r){
        for (int s = 0; s < N_spherical; ++s){
            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nvz; ++k){
                        int idx3 = (i * Nvy + j) * Nvz + k;
                        int idx5 = ((((r) * N_spherical + s) * Nvx + i) * Nvy + j) * Nvz + k;
                        alpha1_times_f_hat[idx5] = fft_scale * alpha1[idx5] * f_hat[idx3];
                        alpha2_times_f_hat[idx5] = fft_scale * std::conj(alpha1[idx5]) * f_hat[idx3];
                    }
                }
            }
        }
    }

    #pragma omp barrier

    // Compute alpha1_times_f = ifft(alpha1_times_f_hat) and 
    // alpha2_times_f = ifft(alpha2_times_f_hat)
    #pragma omp for
    for(int b = 0; b < batch_size; ++b){
        fftw_execute_dft(ifft_plan, 
                        reinterpret_cast<fftw_complex*>(alpha1_times_f_hat + b * grid_size), 
                        reinterpret_cast<fftw_complex*>(alpha1_times_f + b * grid_size));
    }

    #pragma omp for
    for(int b = 0; b < batch_size; ++b){
        fftw_execute_dft(ifft_plan, 
                        reinterpret_cast<fftw_complex*>(alpha2_times_f_hat + b * grid_size), 
                        reinterpret_cast<fftw_complex*>(alpha2_times_f + b * grid_size));
    }
 
    #pragma omp barrier

    // Compute transform_prod = alpha1_times_f * alpha2_times_f
    #pragma omp for collapse(4)
    for (int r = 0; r < N_gl; ++r){
        for (int s = 0; s < N_spherical; ++s){
            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nvz; ++k){
                        int idx5 = ((((r) * N_spherical + s) * Nvx + i) * Nvy + j) * Nvz + k;
                        transform_prod[idx5] = alpha1_times_f[idx5]*alpha2_times_f[idx5];
                    }
                }
            }
        }
    }

    #pragma omp barrier

    // Compute transform_prod_hat = fft(transform_prod)
    #pragma omp for
    for(int b = 0; b < batch_size; ++b){
        fftw_execute_dft(fft_plan, 
                        reinterpret_cast<fftw_complex*>(transform_prod + b * grid_size), 
                        reinterpret_cast<fftw_complex*>(transform_prod_hat + b * grid_size));
    }

    #pragma omp barrier

    // Compute Q_gain_hat
    #pragma omp for collapse(3)
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            for (int k = 0; k < Nvz; ++k){

                int idx3 = (i * Nvy + j) * Nvz + k;
                std::complex<double> sum = 0.0;

                for (int r = 0; r < N_gl; ++r){
                    for (int s = 0; s < N_spherical; ++s){
                   
                        int idx4 = (((r * Nvx + i) * Nvy + j) * Nvz + k);
                        int idx5 = ((((r) * N_spherical + s) * Nvx + i) * Nvy + j) * Nvz + k; 
                        double weight = fft_scale*gl_wts[r]*spherical_wts[s]*std::pow(gl_nodes[r], gamma+2);
                        sum += weight * beta1[idx4] * transform_prod_hat[idx5];
                         
                    }
                }

            Q_gain_hat[idx3] = sum;

            }
        }
    }

/* This uses a custom reducer... not supported by NVIDIA OpenMP compilers

    #pragma omp for collapse(2) reduction(+:Q_gain_hat[:grid_size])
    for (int r = 0; r < N_gl; ++r){
        for (int s = 0; s < N_spherical; ++s){

            double weight = fft_scale*gl_wts[r]*spherical_wts[s]*std::pow(gl_nodes[r], gamma+2);

            #pragma omp simd collapse(3)
            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    for (int k = 0; k < Nvz; ++k){
                        int idx3 = (i * Nvy + j) * Nvz + k;
                        int idx4 = (((r * Nvx + i) * Nvy + j) * Nvz + k);
                        int idx5 = ((((r) * N_spherical + s) * Nvx + i) * Nvy + j) * Nvz + k;
                        Q_gain_hat[idx3] += weight*beta1[idx4]*transform_prod_hat[idx5];
                    }
                }
            }

        }
    }

*/

    // Compute beta2_times_f_hat = beta2 * f_hat
    #pragma omp for collapse(2)
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            #pragma omp simd
            for (int k = 0; k < Nvz; ++k){
                int idx3 = (i * Nvy + j) * Nvz + k;
                beta2_times_f_hat[idx3] = fft_scale*beta2[idx3]*f_hat[idx3];
            }
        }
    }

    #pragma omp barrier

    // Compute Q_gain = ifft(Q_gain_hat)
    #pragma omp single
    fftw_execute_dft(ifft_plan, 
                    reinterpret_cast<fftw_complex*>(Q_gain_hat), 
                    reinterpret_cast<fftw_complex*>(Q_gain));

    // Compute beta2_times_f = ifft(beta2_times_f_hat)
    #pragma omp single
    fftw_execute_dft(ifft_plan, 
        reinterpret_cast<fftw_complex*>(beta2_times_f_hat), 
        reinterpret_cast<fftw_complex*>(beta2_times_f));

    #pragma omp barrier

    // Compute Q = real(Q_gain) - real(Q_loss)
    #pragma omp for collapse(2)
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            #pragma omp simd
            for (int k = 0; k < Nvz; ++k){
                int idx3 = (i * Nvy + j) * Nvz + k;
                std::complex<double> Q_loss = beta2_times_f[idx3]*f[idx3];
                Q[idx3] = Q_gain[idx3].real() - Q_loss.real();
            }
        }
    }




    } // End of parallel region

}; // End of computeCollision


// Class destructor to clean up FFTW resources
BoltzmannOperator<FFTW_Backend>::~BoltzmannOperator() {

    // Clean up FFTW plans
    fftw_destroy_plan(fft_plan);
    fftw_destroy_plan(ifft_plan);

    // Free allocated arrays for the transforms and weights
    fftw_free(alpha1);
    fftw_free(beta1);
    fftw_free(beta2);

    fftw_free(data);
    fftw_free(data_hat);

    fftw_free(f);
    fftw_free(f_hat);

    fftw_free(alpha1_times_f);
    fftw_free(alpha1_times_f_hat);
    
    fftw_free(alpha2_times_f);
    fftw_free(alpha2_times_f_hat);
    
    fftw_free(beta2_times_f);
    fftw_free(beta2_times_f_hat);
    
    fftw_free(transform_prod);
    fftw_free(transform_prod_hat);
    
    fftw_free(Q_gain_hat);
    fftw_free(Q_gain);

}; // End of destructor
