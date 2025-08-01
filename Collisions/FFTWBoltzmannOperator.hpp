#ifndef FFTW_BOLTZMANN_OPERATOR_HPP
#define FFTW_BOLTZMANN_OPERATOR_HPP

#include <omp.h>
#include <fftw3.h>
#include <cmath>
#include <string>
#include <memory>
#include <limits> // Move this to the generic header?
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
class BoltzmannOperator<FFTW_Backend> : public AbstractCollisionOperator {
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

    // Fourier modes for the velocity domain
    std::vector<int> lx, ly, lz;

private:

    // Plans for the transforms
    fftw_plan fft_plan;
    fftw_plan ifft_plan;

    // Weights needed to scale the transforms
    // fftw_complex* alpha1;
    // double* beta1;
    // double* beta2;

    // Generic arrays to create plans for the forward and inverse transforms 
    // fftw_complex* data;
    // fftw_complex* data_hat;

    fftw_complex* f; 
    fftw_complex* f_hat; 
   
    fftw_complex* alpha1_times_f;
    fftw_complex* alpha1_times_f_hat; 
    
    fftw_complex* alpha2_times_f; 
    fftw_complex* alpha2_times_f_hat; 
    
    fftw_complex* beta2_times_f;
    fftw_complex* beta2_times_f_hat; 
    
    fftw_complex* transform_prod; 
    fftw_complex* transform_prod_hat; 

    fftw_complex* Q_gain_hat;
    fftw_complex* Q_gain; 

    fftw_complex* Q_loss_hat;
    fftw_complex* Q_loss; 

};

#endif // FFTW_BOLTZMANN_OPERATOR_HPP
