#ifndef NEURALNETWORKS_H
#define NEURALNETWORKS_H

#include <vector>
#include "denselayer.h"
#include "optimizer.h"

class NeuralNetwork {
    public:
        std::vector<DenseLayer> layers;

        NeuralNetwork(const std::vector<size_t> &layer_sizes, 
                        const std::vector<std::function<double (double)> > &activations,
                        const std::vector<std::function<double (double)> > &activation_derivatives,
                        double learning_rate,
                        std::shared_ptr<SGD> opt
                        );

        Tensor forward(const Tensor &input);
        Tensor lagrange_forward(const Tensor &input, double mass, double gravity);

        void backward(const Tensor &target, double lambda);

        double compute_lagrangian(const Tensor &positions, const Tensor &vel, double mass, double gravity);

        Tensor enforce_lagrangian(const Tensor &positions, const Tensor &vel, const Tensor &acc, double mass, double gravity);

        double compute_physics_loss_with_lagrangian(const Tensor &pos, const Tensor &vel, const Tensor &acc, double mass, double gravity);
};

#endif