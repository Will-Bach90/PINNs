#ifndef NEURALNETWORKS_H
#define NEURALNETWORKS_H

#include <vector>
#include "denselayer.h"

class NeuralNetwork {
    public:
        std::vector<DenseLayer> layers;

        NeuralNetwork(const std::vector<size_t> &layer_sizes, 
                        const std::vector<std::function<double (double)> > &activations,
                        const std::vector<std::function<double (double)> > &activation_derivatives);

        Tensor forward(const Tensor &input);

        void backward(const Tensor &target, double learning_rate);
};

#endif