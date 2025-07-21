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
    //alpha1 = fftw_alloc_complex(batch_size * grid_size); 
    //beta1 = fftw_alloc_real(N_gl * grid_size);
    //beta2 = fftw_alloc_real(grid_size);

    f = fftw_alloc_complex(grid_size);
    f_hat = fftw_alloc_complex(grid_size);
        
    alpha1_times_f = fftw_alloc_complex(batch_size * grid_size);
    alpha1_times_f_hat = fftw_alloc_complex(batch_size * grid_size);

    alpha2_times_f = fftw_alloc_complex(batch_size * grid_size);
    alpha2_times_f_hat = fftw_alloc_complex(batch_size * grid_size);
    
    transform_prod = fftw_alloc_complex(batch_size * grid_size);
    transform_prod_hat = fftw_alloc_complex(batch_size * grid_size);

    Q_gain = fftw_alloc_complex(grid_size);
    Q_gain_hat = fftw_alloc_complex(grid_size);

    beta2_times_f = fftw_alloc_complex(grid_size);
    beta2_times_f_hat = fftw_alloc_complex(grid_size);

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

    fft_plan  = fftw_plan_dft_3d(Nvx, Nvy, Nvz, f, f_hat, FFTW_FORWARD, FFTW_ESTIMATE);
    ifft_plan = fftw_plan_dft_3d(Nvx, Nvy, Nvz, f_hat, f, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Export wisdom immediately after plan creation using the specified filename
    fftw_export_wisdom_to_filename(wisdom_fname.c_str());

}; // End of initialize

/*
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
                        
                        // Define tmp1 = -(pi/(2*L))*gl_nodes[r]*l_dot_sigma
                        double tmp1;
                        tmp1 = -(pi/(2*L)) * gl_nodes[r] * l_dot_sigma;
                        
                        // Compute exp(i*tmp1) = cos(tmp1) + i * sin(tmp1) 
                        alpha1[idx5][0] = std::cos( tmp1 );
                        alpha1[idx5][1] = std::sin( tmp1 );
                    }
                }
            }
        }
    }

    // Compute the real transform weights beta1
    #pragma omp for collapse(3)
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



    } // End of parallel region

}; // End of precomputeTransformWeights
*/


// Compute the collision operator using FFTW
void BoltzmannOperator<FFTW_Backend>::computeCollision(double * Q, const double * f_in) {

    // Retrieve information from the quadrature objects
    int N_gl = gl_quadrature->getNumberOfPoints();
    int N_spherical = spherical_quadrature->getNumberOfPoints();    

    const std::vector<double>& gl_wts = gl_quadrature->getWeights();
    const std::vector<double>& gl_nodes = gl_quadrature->getNodes();
    const std::vector<double>& spherical_wts = spherical_quadrature->getWeights();
    const std::vector<double>& sx = spherical_quadrature->getx();
    const std::vector<double>& sy = spherical_quadrature->gety();
    const std::vector<double>& sz = spherical_quadrature->getz();

    int grid_size = Nvx * Nvy * Nvz; // Total number of grid points
    //int batch_size = N_gl * N_spherical; // Total number of grid_size batches
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
                f[idx3][0] = f_in[idx3];
                f[idx3][1] = 0;
                Q_gain_hat[idx3][0] = 0;
                Q_gain_hat[idx3][1] = 0;
            }
        }
    }

    #pragma omp barrier

    // Compute f_hat = fft(f)
    #pragma omp single
    fftw_execute_dft(fft_plan, f, f_hat);

    #pragma omp barrier

    // Loop over the batches in parallel
    #pragma omp for collapse(2)
    for (int r = 0; r < N_gl; ++r){
        for (int s = 0; s < N_spherical; ++s){

            // Calculate the batch index
            int b = r * N_spherical + s;

            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nvz; ++k){
                        int idx3 = (i * Nvy + j) * Nvz + k;
                        int idx5 = (((b) * Nvx + i) * Nvy + j) * Nvz + k;
                        double l_dot_sigma = lx[i]*sx[s] + ly[j]*sy[s] + lz[k]*sz[s];
                        
                        // Define tmp = -(pi/(2*L))*gl_nodes[r]*l_dot_sigma
                        double tmp;
                        tmp = -(pi/(2*L)) * gl_nodes[r] * l_dot_sigma;

                        // Define a = alpha1 = exp(i*tmp1) = cos(tmp1) + i * sin(tmp1)
                        double a_re = std::cos(tmp);
                        double a_im = std::sin(tmp);
                        double b_re = f_hat[idx3][0];
                        double b_im = f_hat[idx3][1];

                        // alpha1_times_f_hat[idx5] = fft_scale * alpha1[idx5] * f_hat[idx3];
                        alpha1_times_f_hat[idx5][0] = fft_scale * (a_re * b_re - a_im * b_im);
                        alpha1_times_f_hat[idx5][1] = fft_scale * (a_re * b_im + a_im * b_re);

                        // alpha2_times_f_hat[idx5] = fft_scale * conj(alpha1[idx5]) * f_hat[idx3];
                        alpha2_times_f_hat[idx5][0] = fft_scale * (a_re * b_re + a_im * b_im);
                        alpha2_times_f_hat[idx5][1] = fft_scale * (a_re * b_im - a_im * b_re);
                    }
                }
            }

            // Compute alpha1_times_f = ifft(alpha1_times_f_hat) and 
            // alpha2_times_f = ifft(alpha2_times_f_hat)
            fftw_execute_dft(ifft_plan, alpha1_times_f_hat + b * grid_size, alpha1_times_f + b * grid_size);
            fftw_execute_dft(ifft_plan, alpha2_times_f_hat + b * grid_size, alpha2_times_f + b * grid_size);

            // Compute the product transform_prod = alpha1_times_f * alpha2_times_f
            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nvz; ++k){
                        int idx5 = (((b) * Nvx + i) * Nvy + j) * Nvz + k;
                        double a_re = alpha1_times_f[idx5][0];
                        double a_im = alpha1_times_f[idx5][1];
                        double b_re = alpha2_times_f[idx5][0];
                        double b_im = alpha2_times_f[idx5][1];
                        transform_prod[idx5][0] = a_re * b_re - (a_im * b_im);
                        transform_prod[idx5][1] = a_re * b_im + (a_im * b_re);
                    }
                }
            }

            // Compute transform_prod_hat = fft(transform_prod)
            fftw_execute_dft(fft_plan, transform_prod + b * grid_size, transform_prod_hat + b * grid_size); 

            // Compute Q_gain_hat for this batch
            double weight = fft_scale * gl_wts[r] * spherical_wts[s] * std::pow(gl_nodes[r], gamma + 2);

            for (int i = 0; i < Nvx; ++i){
                for (int j = 0; j < Nvy; ++j){
                    for (int k = 0; k < Nvz; ++k){
        
                        int idx3 = (i * Nvy + j) * Nvz + k;
                        int idx5 = (((b) * Nvx + i) * Nvy + j) * Nvz + k;

                        double norm_l = std::sqrt(lx[i]*lx[i] + ly[j]*ly[j] + lz[k]*lz[k]);
                        double beta1 = 4*pi*b_gamma*sincc(pi*gl_nodes[r]*norm_l/(2*L));

                        // Try scaling with atomics first... If this ends up being a bottleneck, we can
                        // switch to a reduction approach using temporaries since Nvidia does not support
                        // complex reductions in OpenMP.
                        #pragma omp atomic
                        Q_gain_hat[idx3][0] += weight * beta1 * transform_prod_hat[idx5][0];
                        #pragma omp atomic
                        Q_gain_hat[idx3][1] += weight * beta1 * transform_prod_hat[idx5][1];
                    }
                }
            }

        }
    }

    #pragma omp barrier

    // Compute beta2_times_f_hat = beta2 * f_hat
    #pragma omp for collapse(3)
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            for (int k = 0; k < Nvz; ++k){

                int idx3 = (i * Nvy + j) * Nvz + k;
                double beta2 = 0.0;
                double norm_l = std::sqrt(lx[i]*lx[i] + ly[j]*ly[j] + lz[k]*lz[k]);

                #pragma omp simd reduction(+:beta2)
                for (int r = 0; r < N_gl; ++r){
                    beta2 += 16*pi*pi*b_gamma*gl_wts[r]*std::pow(gl_nodes[r], gamma+2)*sincc(pi*gl_nodes[r]*norm_l/L);
                }

                beta2_times_f_hat[idx3][0] = fft_scale * beta2 * f_hat[idx3][0];
                beta2_times_f_hat[idx3][1] = fft_scale * beta2 * f_hat[idx3][1];
            }
        }
    }

    #pragma omp barrier

    // Compute Q_gain = ifft(Q_gain_hat)
    #pragma omp single
    fftw_execute_dft(ifft_plan, Q_gain_hat, Q_gain); 

    // Compute beta2_times_f = ifft(beta2_times_f_hat)
    #pragma omp single
    fftw_execute_dft(ifft_plan, beta2_times_f_hat, beta2_times_f); 
 
    #pragma omp barrier

    // Compute Q = real(Q_gain) - real(Q_loss)
    #pragma omp for collapse(2)
    for (int i = 0; i < Nvx; ++i){
        for (int j = 0; j < Nvy; ++j){
            #pragma omp simd
            for (int k = 0; k < Nvz; ++k){
                int idx3 = (i * Nvy + j) * Nvz + k;
                double a_re = beta2_times_f[idx3][0];
                double a_im = beta2_times_f[idx3][1];
                double b_re = f[idx3][0];
                double b_im = f[idx3][1];
                fftw_complex Q_loss;
                Q_loss[0] = a_re * b_re - (a_im * b_im);
                Q_loss[1] = a_re * b_im + (a_im * b_re);
                Q[idx3] = Q_gain[idx3][0] - Q_loss[0];
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
