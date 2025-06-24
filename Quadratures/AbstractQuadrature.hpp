#ifndef ABSTRACT_QUADRATURE_HPP
#define ABSTRACT_QUADRATURE_HPP

#include <vector>
#include <iostream>
#include <stdexcept>

class AbstractQuadrature {
public:
    // Default constructor
    AbstractQuadrature() = default;

    // Virtual destructor for proper cleanup of derived classes
    virtual ~AbstractQuadrature() = default;

    // Getter for weights
    const std::vector<double>& getWeights() const {
        return weights;
    }

    // Getter for nodes
    const std::vector<double>& getNodes() const {
        return nodes;
    }

    // Getter for the number of quadrature points
    int getNumberOfPoints() const {
        return weights.size();
    }

    // Function to print the quadrature information
    virtual void printQuadratureInfo() const {
        std::cout << "Quadrature Weights: ";
        for (const auto& weight : weights) {
            std::cout << weight << " ";
        }
        std::cout << "\nQuadrature Nodes: ";
        for (const auto& node : nodes) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }

protected:
    std::vector<double> weights; // Quadrature weights
    std::vector<double> nodes;   // Quadrature nodes
};

#endif // ABSTRACT_QUADRATURE_HPP