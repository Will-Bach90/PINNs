#include "../include/denselayer.h"

DenseLayer::DenseLayer(size_t input_size, size_t output_size, 
            const std::function<double(double)> &activation_function, 
            const std::function<double(double)> &activation_derivative_function,
            std::shared_ptr<SGD> opt) 
            : weights(input_size, output_size)
            , biases(1, output_size)
            , activation(activation_function)
            , activation_derivative(activation_derivative_function)
            , weight_gradients(input_size, output_size)
            , bias_gradients(1, output_size)
            , optimizer(opt)
            {

                for(size_t i = 0; i < weights.rows(); ++i) {                        // randomly initialize weights
                    for(size_t j =  0; j < weights.cols(); ++j) {
                        weights.data_[i][j] = random_double(-1.0, 1.0);
                    }
                }

                for(size_t j = 0; j < biases.cols(); ++j) {
                    biases.data_[0][j] = random_double(-1.0, 1.0);                  // randomly initialize biases
                }    
            }

Tensor DenseLayer::forward(const Tensor &input) {                                   // forward base through layer: 
    inputs = input;
    Tensor z = (input * weights) + biases;                                          // z = Wx + b
    outputs = z.apply(activation);                                                  // a = sigmoid(z) (for example)
    return outputs;
}

Tensor DenseLayer::backward(const Tensor &gradient, double learning_rate) {         // Backward pass
    // Compute dz = gradient * activation_derivative(outputs)
    Tensor dz(inputs.rows(), weights.cols());                                       
    for (size_t i = 0; i < dz.rows(); ++i) {
        for (size_t j = 0; j < dz.cols(); ++j) {
            dz.data_[i][j] = gradient.data_[i][j] * activation_derivative(outputs.data_[i][j]);         // dz = da * g'(z)
        }
    }

    // Compute weight gradients: dweights = inputs^T * dz
    Tensor dweights(weights.rows(), weights.cols());                                                    
    for (size_t i = 0; i < dweights.rows(); ++i) {
        for (size_t j = 0; j < dweights.cols(); ++j) {
            dweights.data_[i][j] = 0.0; // Reset accumulator
            for (size_t k = 0; k < dz.rows(); ++k) {
                dweights.data_[i][j] += inputs.data_[k][i] * dz.data_[k][j];                            // dw = dz * a^T
            }
        }
    }

    // Compute bias gradients: dbiases = sum(dz over rows)
    Tensor dbiases(1, biases.cols());                                                                   // db = dz
    for (size_t j = 0; j < dbiases.cols(); ++j) {
        dbiases.data_[0][j] = 0.0; // Reset accumulator
        for (size_t i = 0; i < dz.rows(); ++i) {
            dbiases.data_[0][j] += dz.data_[i][j];
        }
    }

    // Update weights and biases
    optimizer->update(weights, dweights, learning_rate);
    optimizer->update_biases(biases, dbiases, learning_rate);
    // for (size_t i = 0; i < weights.rows(); ++i) {
    //     for (size_t j = 0; j < weights.cols(); ++j) {
    //         weights.data_[i][j] -= learning_rate * dweights.data_[i][j];
    //     }
    // }
    // for (size_t j = 0; j < biases.cols(); ++j) {
    //     biases.data_[0][j] -= learning_rate * dbiases.data_[0][j];
    // }

    // Compute gradient to pass to previous layer: dinputs = dz * weights^T
    Tensor dinputs(dz.rows(), weights.rows());
    for (size_t i = 0; i < dinputs.rows(); ++i) {
        for (size_t j = 0; j < dinputs.cols(); ++j) {
            dinputs.data_[i][j] = 0.0; // Reset accumulator
            for (size_t k = 0; k < dz.cols(); ++k) {
                dinputs.data_[i][j] += dz.data_[i][k] * weights.data_[j][k];                // da = W * dz
            }
        }
    }

    return dinputs;
}