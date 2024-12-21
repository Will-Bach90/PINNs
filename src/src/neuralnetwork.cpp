#include "../include/neuralnetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layer_sizes, 
                const std::vector<std::function<double (double)> > &activations,
                const std::vector<std::function<double (double)> > &activation_derivatives) 
                {
                    for(size_t i = 1; i < layer_sizes.size(); ++i) {
                        layers.emplace_back(layer_sizes[i-1], layer_sizes[i], activations[i-1], activation_derivatives[i-1]);
                    }
                }

Tensor NeuralNetwork::forward(const Tensor &input) {
    Tensor output = input;
    for(auto &layer : layers) {
        output = layer.forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const Tensor &target, double learning_rate) {
    Tensor gradient = layers.back().outputs;
    for (size_t i = 0; i < gradient.rows(); ++i) {
        for(size_t j = 0; j < gradient.cols(); ++j) {
            gradient.data_[i][j] -= target.data_[i][j];
        }
    }

    for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
        gradient = it->backward(gradient, learning_rate);
    }
}