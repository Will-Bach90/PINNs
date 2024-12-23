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
