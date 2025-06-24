#ifndef MULTIDIMINDEXER_HPP
#define MULTIDIMINDEXER_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>

template <std::size_t N>
class MultiDimIndexer {
public:

    // Note: In C++20, we can use std::span to avoid the need for std::array
    // and std::vector in the constructor. This would allow for more flexible
    // initialization of dimensions. However, for compatibility with C++11/14,
    // we will stick with std::array and std::vector

    // Constructor to initialize dimensions from an std::array
    MultiDimIndexer(const std::array<std::size_t, N>& dimensions)
        : dims(dimensions) {
        computeStrides();
    }

    // Constructor to initialize dimensions from an initializer list
    MultiDimIndexer(std::initializer_list<std::size_t> dimensions) {
        if (dimensions.size() != N) {
            throw std::invalid_argument("Number of dimensions must match template parameter N.");
        }

        std::copy(dimensions.begin(), dimensions.end(), dims.begin());
        computeStrides();
    }

    // Constructor to initialize dimensions from a std::vector
    MultiDimIndexer(std::vector<std::size_t> dimensions) {
        if (dimensions.size() != N) {
            throw std::invalid_argument("Number of dimensions must match template parameter N.");
        }

        std::copy(dimensions.begin(), dimensions.end(), dims.begin());
        computeStrides();
    }

    ~MultiDimIndexer() = default;

    // Overload the parenthesis operator to compute the linear index
    template <typename... Indices>
    std::size_t operator()(Indices... indices, bool check_bounds = false) const {
        static_assert(sizeof...(Indices) == N, "Number of indices must match dimensions.");
        std::array<std::size_t, N> idx_array{static_cast<std::size_t>(indices)...};

        if (check_bounds) {
            for (std::size_t i = 0; i < N; ++i) {
                if (idx_array[i] >= dims[i]) {
                    throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
                }
            }
        }

        std::size_t index = 0;
        for (std::size_t i = 0; i < N; ++i) {
            index += idx_array[i]*strides[i];
        }
        return index;
    }

    const std::array<std::size_t, N>& getDimensions() const {
        return dims;
    }

    const std::array<std::size_t, N>& getStrides() const {
        return strides;
    }

private:
    std::array<std::size_t, N> dims;    // Dimensions of the multi-dimensional array
    std::array<std::size_t, N> strides; // Strides for each dimension

    // Compute strides for each dimension
    void computeStrides() {
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i) {
            strides[i] = strides[i + 1]*dims[i + 1];
        }
    }
};

#endif // MULTIDIMINDEXER_HPP

