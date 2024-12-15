#include "../include/linear.h"
#include <random>

Linear::Linear(std::size_t in_dim, std::size_t out_dim) 
    : W(in_dim * out_dim), b(out_dim), in_dim(in_dim), out_dim(out_dim)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    for (auto &w_val : W.value) {
        w_val = dist(gen);
    }
    for (auto &b_val : b.value) {
        b_val = dist(gen);
    }
}

Node* Linear::forward(Node* x) {
    return linear_layer(&W, &b, x, out_dim, in_dim);
}

Node* Linear::backward(Node* loss) {
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

Node* Linear::linear_layer(Node* W, Node* b, Node* x, std::size_t output_dim, std::size_t input_dim) {
    Node* y = matmul(W, x, output_dim, input_dim);
    Node* out = add(y, b, output_dim);
    return out;
}