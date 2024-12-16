#ifndef LOSS_H
#define LOSS_H

#include "node.h"

inline Node* mse_loss(Node* pred, double target) {
    Node* out = new Node(1);
    double diff = pred->value[0] - target;
    out->value[0] = 0.5 * diff * diff;

    out->parents = {pred};

    out->backward_op.backward_func = [out,pred,target](const std::vector<double> &dOut) {
        double go = dOut[0];
        double diff = pred->value[0] - target;
        pred->grad_[0] += go * diff; 
    };

    return out;
}

#endif