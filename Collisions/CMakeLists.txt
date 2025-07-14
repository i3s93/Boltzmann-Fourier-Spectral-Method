target_sources(Boltzmann PRIVATE
                AbstractCollisionOperator.hpp
                BoltzmannOperator.hpp
                FFTWBoltzmannOperator.hpp
                FFTWBoltzmannOperator.cpp
                BoltzmannCUDAKernels.hpp
                BoltzmannCUDAKernels.cu
                CUDABoltzmannOperator.hpp
                CUDABoltzmannOperator.cu)
