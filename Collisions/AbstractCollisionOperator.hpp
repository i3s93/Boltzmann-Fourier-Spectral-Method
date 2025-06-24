#ifndef ABSTRACT_COLLISION_OPERATOR_HPP
#define ABSTRACT_COLLISION_OPERATOR_HPP

#include <vector>
#include <string>
#include "../Utilities/constants.hpp"

template <typename Backend>
class AbstractCollisionOperator {
public:
    // Constructor
    AbstractCollisionOperator() = default;

    // Pure virtual function to setup the plans and arrays for the backend
    virtual void initialize() = 0;

    // Pure virtual function to precompute transform weights
    virtual void precomputeTransformWeights() = 0;

    // Pure virtual function to get the backend name
    virtual std::string getBackendName() const = 0;

    // Pure virtual function to compute the collision operator
    virtual void computeCollision(double * Q, const double * f_in) = 0;

    // Pure virtual operator() to call computeCollision
    virtual void operator()(double* Q, const double* f_in) = 0;

    // Virtual destructor for proper cleanup of derived classes
    virtual ~AbstractCollisionOperator() = default;
};

#endif // ABSTRACT_COLLISION_OPERATOR_HPP