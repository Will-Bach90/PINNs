#ifndef LINEAR_H
#define LINEAR_H

// #include "node.h"
#include "ops.h"
#include "autodiff_utils.h"

class Linear {
public:
    Node W;
    Node b;
    std::size_t in_dim;
    std::size_t out_dim;

    Linear(std::size_t in_dim, std::size_t out_dim);

    Node* forward(Node* x);
};

#endif