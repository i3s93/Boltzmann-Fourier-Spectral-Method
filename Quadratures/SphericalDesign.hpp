#ifndef SPHERICAL_DESIGN_HPP
#define SPHERICAL_DESIGN_HPP

#include "AbstractSphericalQuadratures.hpp"
#include "../Utilities/constants.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>

class SphericalDesign : public SphericalQuadrature {
    public:
        // Constructor which takes the number of points in the spherical design
        // and initializes the design
        // This constructor will call the generateNodes and generateWeights methods
        // to populate the nodes and weights vectors
        // The constructor should also handle the case where N is not valid
        // by throwing an exception.
        SphericalDesign(int N);
        ~SphericalDesign() = default; // Default destructor
        
    private:
    int N; // Number of points in the spherical design
    std::string filename; // Filename for the spherical design data

    };
    
    #endif // SPHERICAL_DESIGN_HPP