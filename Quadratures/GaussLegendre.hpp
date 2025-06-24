#ifndef GAUSS_LEGENDRE_QUADRATURE_HPP
#define GAUSS_LEGENDRE_QUADRATURE_HPP

#include "AbstractQuadrature.hpp"
#include <gsl/gsl_integration.h>

class GaussLegendreQuadrature : public AbstractQuadrature {
public:
    // Constructor to initialize Gauss-Legendre quadrature on the interval [a, b]
    GaussLegendreQuadrature(int n_points, double a, double b) {
        weights.resize(n_points);
        nodes.resize(n_points);

        gsl_integration_glfixed_table* table = gsl_integration_glfixed_table_alloc(n_points);
        if (!table) {
            throw std::runtime_error("Failed to allocate GSL integration table.");
        }

        for (int i = 0; i < n_points; ++i) {
            gsl_integration_glfixed_point(a, b, i, &nodes[i], &weights[i], table);
        }

        gsl_integration_glfixed_table_free(table);
    }

    // Override the print function to include Gauss-Legendre-specific information
    void printQuadratureInfo() const override {
        std::cout << "Gauss-Legendre Quadrature:\n";
        AbstractQuadrature::printQuadratureInfo();
    }
};

#endif // GAUSS_LEGENDRE_QUADRATURE_HPP