#ifndef CUDA_BOLTZMANN_OPERATOR_HPP
#define CUDA_BOLTZMANN_OPERATOR_HPP

#include <cmath>
#include <string>
#include <memory>
#include <limits>

#include <omp.h>
#include <cuda_runtime.h>
#include <cutensornet.h>
#include <cufft.h>

#include "BoltzmannOperator.hpp"
#include "BoltzmannCUDAKernels.hpp"
#include "AbstractCollisionOperator.hpp"
#include "../Quadratures/GaussLegendre.hpp"
#include "../Quadratures/SphericalDesign.hpp"

// This function is related to the interaction kernel in the Boltzmann operator and will likely be
// moved outside of this class for better modularity.
template<typename T>
T sincc(T x){
    T eps = std::numeric_limits<T>::epsilon();
    return std::sin(x + eps)/(x + eps);
}

#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
  if( err != CUTENSORNET_STATUS_SUCCESS )                         \
  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};

#define CUFFT_CALL(func)                                                      \
    {                                                                         \
        cufftResult status = (func);                                          \
        if (status != CUFFT_SUCCESS) {                                        \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__     \
                      << " code " << status << " (" << cufftGetErrorString(status) << ")" << std::endl; \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
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

struct CUDA_Backend {};
template <>
class BoltzmannOperator<CUDA_Backend> : public AbstractCollisionOperator<CUDA_Backend> {
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

    // Precomputes the transform weights used in the cuFFT implementation (must be defined)
    void precomputeTransformWeights() override;

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

    // Fourier modes for the velocity domain
    std::vector<int> lx, ly, lz;

private:

    // Plans for the transforms
    cufftHandle plan3d;
    cufftHandle plan3d_batched;

    // Storage for the fast contraction algorithm
    cutensornetHandle_t cutensornet_handle; // Library handle for cuTensorNet
    cutensornetNetworkDescriptor_t descNet; // Network descriptor for the contraction
    cutensornetContractionOptimizerConfig_t optimizerConfig; // Optimizer configuration for the contraction
    cutensornetContractionOptimizerInfo_t optimizerInfo; // Optimizer info for the contraction
    cutensornetWorkspaceDescriptor_t workDesc; // Workspace descriptor for the contraction
    void * workspace; // Workspace (device) for the contraction operation
    cutensornetPlan_t contraction_plan; // Plan for the contraction operation
    cutensornetContractionAutotunePreference_t autotunePref; // Autotune preference for the contraction

    // Device arrays used to evaluate the collision operator
    // We shall use _h to denote arrays stored on the host (CPU)
    // Otherwise, the arrays are stored on the device (GPU)
    Complex * alpha1;
    Complex * beta1;
    double * beta2;

    Complex * radial_term;
    Complex * spherical_wts;

    Complex * f;
    Complex * f_hat;
    
    Complex * alpha1_times_f;
    Complex * alpha1_times_f_hat; 
    
    Complex * alpha2_times_f; 
    Complex * alpha2_times_f_hat; 
    
    Complex * beta2_times_f; 
    Complex * beta2_times_f_hat; 
    
    Complex * transform_prod; 
    Complex * transform_prod_hat;

    Complex * Q_gain_hat; 
    Complex * Q_gain;

    Complex * Q_loss_hat; 
    Complex * Q_loss;

};

#endif // CUDA_BOLTZMANN_OPERATOR_HPP
