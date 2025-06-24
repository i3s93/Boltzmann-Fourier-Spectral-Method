#ifndef BOLTZMANN_OPERATOR_HPP
#define BOLTZMANN_OPERATOR_HPP

// Define a generic Boltzmann operator class template
// The user should specialize this class for different backends (e.g., FFTW, CUDA, etc.)
// and leave the generic undefined.
template <typename Backend>
class BoltzmannOperator;

#endif // BOLTZMANN_OPERATOR_HPP