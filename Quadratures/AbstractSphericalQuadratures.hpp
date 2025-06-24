#ifndef ABSTRACT_SPHERICAL_QUADRATURES_HPP
#define ABSTRACT_SPHERICAL_QUADRATURES_HPP


#include <vector>
#include <iostream>

// An abstract base class for spherical quadrature rules
// This class provides a common interface for all spherical quadrature rules
// and contains the common data members for weights and nodes as well as methods to access them.
class SphericalQuadrature {
    public:

        // Default constructor
        SphericalQuadrature() = default;

        // Virtual destructor for proper cleanup of derived classes
        virtual ~SphericalQuadrature() = default;
    
        // Getter for weights
        const std::vector<double>& getWeights() const {
            return weights;
        }
    
        // Getter for nodes (stored as x, y, z coordinates)
        const std::vector<double>& getx() const {
            return x;
        }
    
        const std::vector<double>& gety() const {
            return y;
        }
    
        const std::vector<double>& getz() const {
            return z;
        }


        // Getter for the number of quadrature points
        int getNumberOfPoints() const {
            return weights.size();
        }
    
        // Function to print the quadrature information
        void printQuadratureInfo() const {
            std::cout << "Quadrature Weights: ";
            for (const auto& weight : weights) {
                std::cout << weight << " ";
            }
            std::cout << "\nQuadrature Nodes (x, y, z):\n";
            for (size_t i = 0; i < x.size(); ++i) {
                std::cout << "(" << x[i] << ", " << y[i] << ", " << z[i] << ")\n";
            }
        }
    
    protected:
        std::vector<double> weights; // Quadrature weights
        std::vector<double> x;       // x-coordinates of the quadrature points
        std::vector<double> y;       // y-coordinates of the quadrature points
        std::vector<double> z;       // z-coordinates of the quadrature points
    };
    
    #endif // ABSTRACT_SPHERICAL_QUADRATURES_HPP