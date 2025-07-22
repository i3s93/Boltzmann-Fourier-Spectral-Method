#ifndef CUDA_BOLTZMANN_OPERATOR_HPP
#define CUDA_BOLTZMANN_OPERATOR_HPP

#include <cmath>
#include <string>
#include <memory>

#include <omp.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "BoltzmannOperator.hpp"
#include "BoltzmannCUDAKernels.hpp"
#include "AbstractCollisionOperator.hpp"
#include "../Quadratures/GaussLegendre.hpp"
#include "../Quadratures/SphericalDesign.hpp"
#include "../Utilities/constants.hpp"

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

std::string cufftGetErrorString(cufftResult error);

struct CUDA_Backend {};
template <>
class BoltzmannOperator<CUDA_Backend> : public AbstractCollisionOperator {
public:
    // Constructor which accepts Gauss-Legendre and spherical quadrature objects as
    // well as the number of grid points in each velocity direction
    BoltzmannOperator(std::shared_ptr<GaussLegendreQuadrature> gl_quadrature,
        std::shared_ptr<SphericalQuadrature> spherical_quadrature, 
        int Nvx, int Nvy, int Nvz, double gamma, double b_gamma, double L)
        : gl_quadrature(gl_quadrature),
        spherical_quadrature(spherical_quadrature),
        Nvx(Nvx), Nvy(Nvy), Nvz(Nvz), 
        gamma(gamma), b_gamma(b_gamma), L(L) {}

    // Method to setup the CUDA resources (must be defined)
    void initialize() override;

    // Returns the name of the backend being used
    std::string getBackendName() const override {
        return "CUDA";
    }

    // Implement the cuFFT-based collision operator computation (must define)
    void computeCollision(double * Q, const double * f_in) override;

    // Override operator() to call computeCollision
    void operator()(double * Q, const double * f_in) {
        computeCollision(Q, f_in);
    }

    // Destructor that cleans up CUDA resources
    ~BoltzmannOperator() override;

protected:

    // Grid dimensions for the velocity domain
    const int Nvx, Nvy, Nvz;

    // Parameters for the Boltzmann operator
    const double gamma; // Power of the velocity in the interaction kernel
    const double b_gamma; // Coefficient in the interaction kernel
    const double L; // Length scale for the velocity domain

    // Shared pointers to quadrature objects
    const std::shared_ptr<GaussLegendreQuadrature> gl_quadrature;
    const std::shared_ptr<SphericalQuadrature> spherical_quadrature;

private:

    // Plans for the transforms
    cufftHandle plan3d;
    cufftHandle plan3d_batched;

    // Device arrays used to evaluate the collision operator
    // We shall use _h to denote arrays stored on the host (CPU)
    // Otherwise, the arrays are stored on the device (GPU)
    double * gl_wts;
    double * gl_nodes;

    double * spherical_wts;
    double * sx;
    double * sy;
    double * sz;

    int * lx;
    int * ly;
    int * lz;

    cuDoubleComplex * f;
    cuDoubleComplex * f_hat;
    
    cuDoubleComplex * alpha1_times_f;
    cuDoubleComplex * alpha1_times_f_hat; 
    
    cuDoubleComplex * alpha2_times_f; 
    cuDoubleComplex * alpha2_times_f_hat; 
    
    cuDoubleComplex * beta2_times_f; 
    cuDoubleComplex * beta2_times_f_hat; 
    
    cuDoubleComplex * transform_prod; 
    cuDoubleComplex * transform_prod_hat;

    cuDoubleComplex * Q_gain_hat; 
    cuDoubleComplex * Q_gain;

    cuDoubleComplex * Q_loss_hat; 
    cuDoubleComplex * Q_loss;

};

#endif // CUDA_BOLTZMANN_OPERATOR_HPP
