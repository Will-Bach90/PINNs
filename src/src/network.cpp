#include "../include/network.h"
#include <cassert>
#include <cmath>
#include <random>

Network::Network(std::size_t input_dim, std::size_t hidden_dim, std::size_t output_dim)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim)
{
    W1_.resize(input_dim_ * hidden_dim_);
    b1_.resize(hidden_dim_);
    W2_.resize(hidden_dim_ * output_dim_);
    b2_.resize(output_dim_);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (std::size_t i = 0; i < W1_.size(); i++) W1_[i] = dist(gen);
    for (std::size_t i = 0; i < b1_.size(); i++) b1_[i] = dist(gen);
    for (std::size_t i = 0; i < W2_.size(); i++) W2_[i] = dist(gen);
    for (std::size_t i = 0; i < b2_.size(); i++) b2_[i] = dist(gen);
}

Tensor Network::forward(const Tensor &input) {
    assert(input.size() == input_dim_);

    Tensor hidden(hidden_dim_);
    for (std::size_t j = 0; j < hidden_dim_; j++) {
        double sum = b1_[j];
        for (std::size_t i = 0; i < input_dim_; i++) {
            sum += W1_[j * input_dim_ + i] * input[i];
        }
        hidden[j] = activation_(sum);
    }

    Tensor output(output_dim_);
    for (std::size_t k = 0; k < output_dim_; k++) {
        double sum = b2_[k];
        for (std::size_t j = 0; j < hidden_dim_; j++) {
            sum += W2_[k * hidden_dim_ + j] * hidden[j];
        }
        output[k] = sum;
    }

    return output;
}
