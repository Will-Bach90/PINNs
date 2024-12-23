#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <functional>
#include <vector>
#include <cmath>
#include "tensor.h"
#include "utils.h"

class DenseLayer {
    public:
        Tensor weights;
        Tensor biases;
        Tensor inputs;
        Tensor outputs;
        Tensor weight_gradients;
        Tensor bias_gradients;
        std::function<double (double) > activation;
        std::function<double(double)> activation_derivative;

        DenseLayer(size_t input_size, size_t output_size, 
                    const std::function<double(double)> &activation_function, 
                    const std::function<double(double)> &activation_derivative_function);
        
        Tensor forward(const Tensor &input);
        Tensor backward(const Tensor &gradient, double learning_rate);
};

#endif