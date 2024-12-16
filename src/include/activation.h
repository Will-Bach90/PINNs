#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "node.h"

// ==============================================================================
// TANH
inline Node* tanh_op(Node* input) {
    Node* out = new Node(input->value.size());
    for (std::size_t i = 0; i < input->value.size(); i++) {
        double val = input->value[i];
        double t = std::tanh(val);
        out->value[i] = t;
    }

    out->parents = {input};

    out->backward_op.backward_func = [out, input](const std::vector<double> &dOut) {
        for (std::size_t i = 0; i < out->value.size(); i++) {
            double t = out->value[i]; 
            double dt = (1.0 - t * t) * dOut[i]; 
            input->grad_[i] += dt;
        }
    };
    return out;
}

// ===============================================================================
// ReLU
inline Node* relu_op(Node* input) {
    Node* out = new Node(input->value.size());
    for (std::size_t i = 0; i < input->value.size(); i++) {
        double val = input->value[i];
        out->value[i] = (val > 0.0) ? val : 0.0;
    }

    out->parents = {input};

    out->backward_op.backward_func = [out,input](const std::vector<double> &dOut) {
        for (std::size_t i = 0; i < out->value.size(); i++) {
            double x_in = input->value[i];
            input->grad_[i] += (x_in > 0.0) ? dOut[i] : 0.0;
        }
    };

    return out;
}
// ===============================================================================
// Sigmoid
inline Node* sigmoid_op(Node* input) {
    Node* out = new Node(input->value.size());
    for (std::size_t i = 0; i < input->value.size(); i++) {
        double val = input->value[i];
        double s = 1.0 / (1.0 + std::exp(-val));
        out->value[i] = s;
    }

    out->parents = {input};

    out->backward_op.backward_func = [out,input](const std::vector<double> &dOut) {
        for (std::size_t i = 0; i < out->value.size(); i++) {
            double s = out->value[i];
            double ds = s * (1.0 - s) * dOut[i];
            input->grad_[i] += ds;
        }
    };

    return out;
}

#endif