#ifndef MLP_H
#define MLP_H

#include <vector>
#include "linear.h"

class MLP {
public:
    std::vector<Linear> layers;

    MLP(const std::vector<std::size_t>& dims);

    Node* forward(Node* x);
};

#endif