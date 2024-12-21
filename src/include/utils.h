#include <cmath>
#include <functional>
#include <vector>
#include <random>

namespace Utils {
    double sigmoid(double x) {
        return 1.0/(1.0 + std::exp(-x));    // 1/(1 + e^(-x))
    }

    double sigmoid_derivative(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);           // 1/(1 + e^(-x))  - (1/(1 + e^(-x)))^2
    }

    double relu(double x) {
        return std::max(0.0, x);
    }

    double relu_derivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

    double random_double(double min, double max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }

}