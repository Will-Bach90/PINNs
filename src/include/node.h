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
    Node(std::size_t = 1);

    std::vector<double> value; 
    
    std::vector<double> grad_;  
    
    BackwardOp backward_op;

    std::vector<Node*> parents;

    void zeroGrad() {
        for (auto &g : grad_)  
        {
            g = 0.0; 
        }  
    }

private:
    std::size_t size_;

};

inline void zeroGrad(std::vector<Node*>& nodes) {
    for (auto* n : nodes) {
        n->zeroGrad();
    }
}

#endif