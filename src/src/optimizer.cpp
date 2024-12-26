#include "../include/optimizer.h"

SGD::SGD() {}

void SGD::update(Tensor &weights, const Tensor &gradients, double learning_rate) {
    for(size_t i = 0; i < weights.rows(); ++i) {
        for(size_t j = 0; j < weights.cols(); ++j) {
            weights.data_[i][j] -= learning_rate * gradients.data_[i][j];
        }
    }
}

void SGD::update_biases(Tensor &biases, const Tensor &gradient, double learning_rate) {
    for(size_t i = 0; i < biases.rows(); ++i) {
        for(size_t j = 0; j < biases.cols(); ++j) {
            biases.data_[i][j] -= learning_rate * gradient.data_[i][j];
        }
    }
}