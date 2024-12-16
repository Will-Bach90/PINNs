#ifndef OPS_H
#define OPS_H

#include "node.h"
#include <iostream>
#include <cassert>
#include <stack>

inline void backward(Node* loss) {
    loss->grad_[0] = 1.0;

    std::stack<Node*> stack;
    stack.push(loss);

    while(!stack.empty()) {
        Node* current = stack.top();
        stack.pop();

        if (current->backward_op.backward_func) {
            current->backward_op.backward_func(current->grad_);
        }

        for (auto* p : current->parents) {
            stack.push(p);
        }
    }
}

inline Node* matmul(Node* W, Node* x, std::size_t output_dim, std::size_t input_dim) {
    Node* out = new Node(output_dim);
    for(std::size_t i = 0; i < output_dim; i++) {
        double sum = 0.0;
        for(std::size_t j = 0; j < input_dim; j++) {
            sum += W->value[i * input_dim + j] * x->value[j];
        }
        out->value[i] = sum;
    }
    out->parents = {W, x};
    out->backward_op.backward_func = [out, W, x, output_dim, input_dim](const std::vector<double>& dOut) {
        for(std::size_t i = 0; i < output_dim; i++) {
            double go = dOut[i];
            for(std::size_t j = 0; j < input_dim; j++) {
                W->grad_[i * input_dim + j] += go * x->value[j];
                x->grad_[j] += go * W->value[i * input_dim + j];
            }
        }
    };

    return out;
}

inline Node* add(Node* y, Node* b, std::size_t output_dim) {
    Node* out = new Node(output_dim);
    double sum = 0.0;
    for (std::size_t i = 0; i < output_dim; i++) {
        out->value[i] = y->value[i] + b->value[i];
    }
    out->parents = {y, b};

    out->backward_op.backward_func = [out, y, b, output_dim](const std::vector<double>& dOut) {
        for (std::size_t i = 0; i < output_dim; i++) {
            y->grad_[i] += dOut[i];
            b->grad_[i] += dOut[i];
        }
    };
    return out;
}


#endif