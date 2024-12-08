#ifndef NODE_H
#define NODE_H

#include <vector>
#include <functional>

class Node;

struct BackwardOp {
    std::function<void(const std::vector<double>&)> backward_func;
};

class Node {
public:
    std::vector<double> value; 
    
    std::vector<double> grad;  
    
    BackwardOp backward_op;

    std::vector<Node*> parents;
    
    Node(std::size_t size=1) : value(size,0.0), grad(size,0.0) {}

    void zeroGrad() {
        for (auto &g : grad) g = 0.0;
    }
};

inline void zeroGrad(std::vector<Node*>& nodes) {
    for (auto* n : nodes) {
        n->zeroGrad();
    }
}

#endif