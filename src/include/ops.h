#ifndef OPS_H
#define OPS_H

#include "node.h"
#include <cassert>


inline Node* linear(Node* W, Node* x, Node* b) {
    assert(W->value.size() == x->value.size());
    assert(b->value.size() == 1);

    Node* out = new Node(1);
    double sum = b->value[0];
    for (std::size_t i = 0; i < x->value.size(); i++) {
        sum += W->value[i] * x->value[i];
    }
    out->value[0] = sum;

    out->parents = {W, x, b};

    out->backward_op.backward_func = [out,W,x,b](const std::vector<double> &dOut) {
        double go = dOut[0];
        for (std::size_t i = 0; i < W->value.size(); i++) {
            W->grad[i] += go * x->value[i];
        }
        for (std::size_t i = 0; i < x->value.size(); i++) {
            x->grad[i] += go * W->value[i];
        }
        b->grad[0] += go;
    };

    return out;
}

inline Node* mse_loss(Node* pred, double target) {
    Node* out = new Node(1);
    double diff = pred->value[0] - target;
    out->value[0] = 0.5 * diff * diff;

    out->parents = {pred};

    out->backward_op.backward_func = [out,pred,target](const std::vector<double> &dOut) {
        double go = dOut[0];
        double diff = pred->value[0] - target;
        pred->grad[0] += go * diff; 
    };

    return out;
}


#endif