// #ifndef NETWORK_H
// #define NETWORK_H

// #include "tensor.h"
// #include <vector>
// #include <functional>
// #include <cmath>

// inline double tanh_activation(double x) {
//     return std::tanh(x);
// }

// class Network {
// public:
//     Network(std::size_t input_dim, std::size_t hidden_dim, std::size_t output_dim);

//     Tensor forward(const Tensor &input);

//     Tensor& getWeights1() { return W1_; }
//     Tensor& getBias1()    { return b1_; }
//     Tensor& getWeights2() { return W2_; }
//     Tensor& getBias2()    { return b2_; }

//     std::size_t inputDim() const { return input_dim_; }
//     std::size_t outputDim() const { return output_dim_; }

// private:
//     std::size_t input_dim_;
//     std::size_t hidden_dim_;
//     std::size_t output_dim_;

//     // W1: input_dim_ * hidden_dim_
//     // b1: hidden_dim_
//     // W2: hidden_dim_ * output_dim_
//     // b2: output_dim_
//     Tensor W1_;
//     Tensor b1_;
//     Tensor W2_;
//     Tensor b2_;

//     std::function<double(double)> activation_ = tanh_activation;

// };

// #endif