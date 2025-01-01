#include "../include/neuralnetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layer_sizes, 
                const std::vector<std::function<double (double)> > &activations,
                const std::vector<std::function<double (double)> > &activation_derivatives,
                double learning_rate,
                std::shared_ptr<SGD> opt
                ) 
                {
                    for(size_t i = 1; i < layer_sizes.size(); ++i) {
                        layers.emplace_back(layer_sizes[i-1], layer_sizes[i], activations[i-1], activation_derivatives[i-1], opt);
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
        // optimizer.update(it->weights, it->weight_gradients);
        // optimizer.update_biases(it->biases, it->bias_gradients);
    }
}

double NeuralNetwork::compute_lagrangian(const Tensor &positions, const Tensor &vel, double mass, double gravity) {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;

    for(size_t i = 0; i < positions.rows(); ++i) {
        double vx = vel.data_[i][0];
        double vy = vel.data_[i][1];
        double vz = vel.data_[i][2];

        // T = 0.5 * m * (vx^2 + vy^2 + vz^2)
        kinetic_energy += 0.5 * mass * (std::pow(vx, 2) + std::pow(vy, 2) + std::pow(vz, 2));

        // V = m * g * z
        double z = positions.data_[i][2];
        potential_energy += mass * gravity * z;
    }

    return kinetic_energy - potential_energy;       // L = T - V
}

Tensor NeuralNetwork::enforce_lagrangian(const Tensor &positions, const Tensor &vel, const Tensor &acc, double mass, double gravity) {
    Tensor lagrangian(positions.rows(), 1);

    for(size_t i = 0; i < positions.rows(); ++i) {
        double x = positions.data_[i][0];
        double y = positions.data_[i][1];
        double z = positions.data_[i][2];

        double vx = vel.data_[i][0];
        double vy = vel.data_[i][1];
        double vz = vel.data_[i][2];

        double ax = acc.data_[i][0];
        double ay = acc.data_[i][1];
        double az = acc.data_[i][2];

        // euler lagrange
        double lagrangian_x = mass * ax - vy;
        double lagrangian_y = mass * ay - vy;
        double lagrangian_z = mass * (az + gravity) - vz;

        lagrangian.data_[i][0] = std::pow(lagrangian_x, 2) + std::pow(lagrangian_y, 2) + std::pow(lagrangian_z, 2);

    }

    return lagrangian;
}