target_sources(Boltzmann PRIVATE
                AbstractQuadrature.hpp
                GaussLegendre.hpp
                AbstractSphericalQuadratures.hpp
                SphericalDesign.hpp
                SphericalDesign.cpp)

find_package(GSL REQUIRED)

target_link_libraries(Boltzmann PUBLIC GSL::gsl GSL::gslcblas)