#include <cmath>
#include <functional>
#include <vector>
#include <random>
#include <numeric>
#include "tensor.h"

inline double sigmoid(double x) {
    return 1.0/(1.0 + std::exp(-x));    // 1/(1 + e^(-x))
}

inline double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);           // 1/(1 + e^(-x))  - (1/(1 + e^(-x)))^2
}

inline double relu(double x) {
    return std::max(0.0, x);
}

inline double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

inline double tanh(double x) {
    return std::tanh(x);
}

inline double tanh_derivative(double x) {
    return (1 - std::pow(std::tanh(x), 2));
}

inline double l2_regularization(const Tensor &weights, double lambda) {
    double sum = 0.0;
    for(const auto &row : weights.data_) {
        sum += std::accumulate(row.begin(), row.end(), 0.0, [](double acc, double val) {
            return acc + val * val;
        });
    }
    return lambda * sum;
}

inline double random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

inline std::vector<double> normalize(const std::vector<double> &data) {
    double max_val = *std::max_element(data.begin(), data.end());
    double min_val = *std::min_element(data.begin(), data.end());
    std::vector<double> normalized_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        normalized_data[i] = (data[i] - min_val) / (max_val - min_val);
    }
    return normalized_data;
}

inline std::vector<std::vector<double > > normalize_2d(const std::vector<std::vector<double > > &data) {
    std::vector<std::vector<double > > normalized_data(data.size(), std::vector<double>(data[0].size()));
    for (size_t j = 0; j < data[0].size(); ++j) {
        double max_val = -std::numeric_limits<double>::infinity();
        double min_val = std::numeric_limits<double>::infinity();
        for (const auto &row : data) {
            max_val = std::max(max_val, row[j]);
            min_val = std::min(min_val, row[j]);
        }
        for (size_t i = 0; i < data.size(); ++i) {
            normalized_data[i][j] = (data[i][j] - min_val) / (max_val - min_val);
        }
    }
    return normalized_data;
}