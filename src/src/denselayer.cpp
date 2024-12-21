#include "../include/denselayer.h"
#include "../include/utils.h"

DenseLayer::DenseLayer(size_t input_size, size_t output_size, 
            const std::function<double(double)> &activation_function, 
            const std::function<double(double)> &activation_derivative_function) 
            : weights(input_size, output_size)
            , biases(1, output_size)
            , activation(activation_function)
            , activation_derivative(activation_derivative_function)
            {

                for(size_t i = 0; i < weights.rows(); ++i) {
                    for(size_t j =  0; j < weights.cols(); ++j) {
                        weights.data_[i][j] = Utils::random_double(-1.0, 1.0);
                    }
                }

                for(size_t j = 0; j < biases.cols(); ++j) {
                    biases.data_[0][j] = Utils::random_double(-1.0, 1.0);
                }    
            }

Tensor DenseLayer::forward(const Tensor &input) {
    inputs = input;
    Tensor z = (input * weights) + biases;
    outputs = z.apply(activation);
    return outputs;
}

Tensor DenseLayer::backward(const Tensor &gradient, double learning_rate) {
    Tensor dz = gradient.apply(activation_derivative);
    Tensor dweights(inputs.rows(), weights.cols());
    for(size_t i = 0; i < inputs.rows(); ++i) {
        for(size_t j = 0; j < weights.cols(); ++j) {
            for(size_t k = 0; k < inputs.cols(); ++k) {
                dweights.data_[k][j] += inputs.data_[i][k] * dz.data_[i][j];
            }
        }
    }

    // gradient descent
    Tensor dbiases(1, biases.cols());
    for(size_t i = 0; i < weights.rows(); ++i) {
        for(size_t j = 0; j < weights.cols(); ++j) {
            weights.data_[i][j] -= learning_rate * dweights.data_[i][j];
        }
    }
    for(size_t i = 0; i < biases.rows(); ++i) {
        for(size_t j = 0; j < biases.cols(); ++j) {
            biases.data_[i][j] -= learning_rate * dbiases.data_[i][j];
        }
    }

    Tensor dinputs(dz.rows(), weights.rows());
    for(size_t i = 0; i < weights.rows(); ++i) {
        for(size_t j = 0; j < dz.rows(); ++j) {
            for(size_t k = 0; k < dz.cols(); ++k) {
                dinputs.data_[j][i] += dz.data_[j][k] * weights.data_[i][k];
            }
        }
    }

    return dinputs;
}