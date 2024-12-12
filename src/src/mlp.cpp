#include "../include/mlp.h"

MLP::MLP(const std::vector<std::size_t>& dims) {
    for (size_t i = 0; i < dims.size()-1; i++) {
        layers.emplace_back(dims[i], dims[i+1]);
    }
}

Node* MLP::forward(Node* x) {
    Node* out = x;
    for (size_t i = 0; i < layers.size(); i++) {
        out = layers[i].forward(out);
        if (i < layers.size()-1) {
            out = relu_op(out);
        }
    }
    return out;
}