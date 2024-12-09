#ifndef AUTODIFF_UTILS_H
#define AUTODIFF_UTILS_H

#include "node.h"
#include <stack>
#include <unordered_set>

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

#endif