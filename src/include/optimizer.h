#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

class SGD {
public:

    double learning_rate;

    SGD(double lr);

    void update(Tensor &weights, const Tensor &gradients);

    void update_biases(Tensor &biases, const Tensor &gradient);

};


#endif