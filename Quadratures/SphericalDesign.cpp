#include "SphericalDesign.hpp"

// Implementation of the SphericalDesign constructor which initializes the
// spherical design rule based on the number of points N. Here, N should be
// a positive integer corresponding to one of the predefined spherical design rules.
SphericalDesign::SphericalDesign(int N) : N(N) {
    if (N <= 0) {
        throw std::invalid_argument("Number of points N must be a positive integer");
    }
    // Initialize the filename based on N
    // TO-DO: Remove the hard coded path... and replace it with something more robust
    switch (N) {
        case 6:   filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss003.006.txt"; break;
        case 12:  filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss005.012.txt"; break;
        case 32:  filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss007.032.txt"; break;
        case 48:  filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss009.048.txt"; break;
        case 70:  filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss011.070.txt"; break;
        case 94:  filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss013.094.txt"; break;
        case 120: filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss015.120.txt"; break;
        case 156: filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss017.156.txt"; break;
        case 192: filename = "/global/homes/w/wsands/Projects/Boltzmann-Fourier-Spectral-Method/Quadratures/ss019.192.txt"; break;
        default:
            throw std::invalid_argument("Invalid value of N");
    }

    // Read the spherical design data from the file
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    std::istringstream iss;
    double x_node, y_node, z_node;

    // Read each line of the file and extract the values on each of the lines
    while (std::getline(file, line)) {
        iss.clear();
        iss.str(line);
        iss >> x_node >> y_node >> z_node;

        x.push_back(x_node);
        y.push_back(y_node);
        z.push_back(z_node);
    }
    // Generate weights (all weights are equal to 4*pi/N)
    weights.resize(N, (4*pi) / N);

}

// Destructor is defaulted, no need to implement it
