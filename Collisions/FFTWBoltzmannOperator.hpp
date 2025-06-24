#ifndef FFTW_BOLTZMANN_OPERATOR_HPP
#define FFTW_BOLTZMANN_OPERATOR_HPP

#include <omp.h>
#include <fftw3.h>
#include <cmath>
#include <complex>
#include <string>
#include <memory>
#include "BoltzmannOperator.hpp"
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

struct FFTW_Backend {};

template <>
class BoltzmannOperator<FFTW_Backend> : public AbstractCollisionOperator<FFTW_Backend> {
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

    // Set the filename for the FFTW3 wisdom file if one is available
    void setWisdomFileName(const std::string& filename) {
        wisdom_fname = filename;
    }

    // Method to setup the FFTW3 plans and arrays (must be defined)
    void initialize() override;

    // Precomputes the transform weights used in the FFTW3 implementation (must be defined)
    void precomputeTransformWeights() override;

    // Returns the name of the backend being used
    std::string getBackendName() const override {
        return "FFTW";
    }

    // Implement the FFTW3-based collision operator computation (must define)
    void computeCollision(double * Q, const double * f_in) override;

    // Override operator() to call computeCollision
    void operator()(double * Q, const double * f_in) {
        computeCollision(Q, f_in);
    }

    // Destructor that cleans up FFTW resources
    ~BoltzmannOperator() override;

protected:

    // Grid dimensions for the velocity domain
    const int Nvx, Nvy, Nvz;

    // Parameters for the Boltzmann operator
    const double gamma; // Power of the velocity in the interaction kernel
    const double b_gamma; // Coefficient in the interaction kernel
    const double L; // Length scale for the velocity domain

    // Name of the wisdom file for FFTW3
    // This file is used to store and retrieve FFTW wisdom for performance optimization
    // The user can set this filename to a custom value. Otherwise, it defaults to "fftw_wisdom.dat"
    std::string wisdom_fname = "fftw_wisdom.dat";

    // Shared pointers to quadrature objects
    const std::shared_ptr<GaussLegendreQuadrature> gl_quadrature;
    const std::shared_ptr<SphericalQuadrature> spherical_quadrature;

private:

    // Scaling for the FFTs for normalization
    double fft_scale;

    // Weights needed for the Fourier transforms
    std::complex<double>* alpha1;
    double* beta1;
    double* beta2;

    // We will store optimized plans for a 3D dataset
    fftw_plan forward_plan;
    fftw_plan backward_plan;

    // Data pointers (place-holders) for a single optimized FFT/iFFT
    std::complex<double>* data_in;
    std::complex<double>* data_out;

    std::complex<double>* f; // This will eventually be removed...
    std::complex<double>* f_hat; // FFTW array for f_hat
    
    std::complex<double>* alpha1_times_f; // FFTW array for alpha1*f
    std::complex<double>* alpha1_times_f_hat; // FFTW array for alpha1*f_hat
    
    std::complex<double>* alpha2_times_f; // FFTW array for alpha2*f
    std::complex<double>* alpha2_times_f_hat; // FFTW array for alpha2*f_hat
    
    std::complex<double>* beta2_times_f; // FFTW array for beta2*f
    std::complex<double>* beta2_times_f_hat; // FFTW array for beta2*f_hat
    
    std::complex<double>* transform_prod; // FFTW array for the product of transforms
    std::complex<double>* transform_prod_hat; // FFTW array for the product of transforms

    std::complex<double>* Q_gain_hat; // FFTW array for Q_gain_hat
    std::complex<double>* Q_gain; // FFTW array for Q_gain

    std::complex<double>* Q_loss_hat; // FFTW array for Q_loss_hat
    std::complex<double>* Q_loss; // FFTW array for Q_loss

};

#endif // FFTW_BOLTZMANN_OPERATOR_HPP