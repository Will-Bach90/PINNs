#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

class SGD {
public:

    SGD();

    void update(Tensor &weights, const Tensor &gradients, double learning_rate);

    void update_biases(Tensor &biases, const Tensor &gradient, double learning_rate);

};


#endif