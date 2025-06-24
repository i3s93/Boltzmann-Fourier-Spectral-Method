#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

template<typename T>
T min(const std::vector<T> &data) {
    T min_val = data[0];
    for (const auto &item : data) {
        if (item < min_val) {
            min_val = item;
        }
    }
    return min_val;
}

template<typename T>
T max(const std::vector<T> &data) {
    T max_val = data[0];
    for (const auto &item : data) {
        if (item > max_val) {
            max_val = item;
        }
    }
    return max_val;
}

template<typename T>
T mean(const std::vector<T> &data) {
    T sum = 0;
    for (const auto &item : data) {
        sum += item;
    }
    return sum / static_cast<T>(data.size());
}

template<typename T>
T stdev(const std::vector<T> &data) {
    T mean_val = mean(data);
    T sum_squared_diff = 0;
    for (const auto &item : data) {
        sum_squared_diff += std::pow(item - mean_val, 2);
    }
    T denom = static_cast<T>(std::max(1, static_cast<int>(data.size() - 1)));
    return std::sqrt(sum_squared_diff / denom);
}

template<typename T>
void print_stats_summary(const std::string &device_name, const std::vector<T> &data){
    std::cout << "\nRun statistics for " << device_name << std::endl;
    std::cout << "Total number of samples taken: " << data.size() << std::endl;
    std::cout << std::scientific << std::setprecision(8) << "Mean runtime (ms): " << mean(data) << "\n";
    std::cout << std::scientific << std::setprecision(8) << "Min runtime (ms): " << min(data) << "\n";
    std::cout << std::scientific << std::setprecision(8) << "Max runtime (ms): " << max(data) << "\n";
    std::cout << std::scientific << std::setprecision(8) << "stdev: " << stdev(data) << "\n";
    std::cout << "\n";
    return;
}

#endif // STATISTICS_HPP
