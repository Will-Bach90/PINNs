#include "../include/node.h"
#include "../include/ops.h"
#include "../include/autodiff_utils.h"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

std::size_t input_dim = 1;
std::size_t hidden_dim = 10;
std::size_t output_dim = 1;

Node W1(hidden_dim);
Node b1(hidden_dim);
Node W2(output_dim * hidden_dim);
Node b2(output_dim);

Node* mlp_forward(Node* x) {
    Node* out1 = new Node(hidden_dim);
    for (std::size_t j = 0; j < hidden_dim; j++) {
        out1->value[j] = W1.value[j] * x->value[0] + b1.value[j];
    }
    out1->parents = {&W1, x, &b1};
    out1->backward_op.backward_func = [out1](const std::vector<double> &dOut){
        Node* W1ptr = out1->parents[0];
        Node* xptr = out1->parents[1];
        Node* b1ptr = out1->parents[2];
        for (std::size_t j = 0; j < out1->value.size(); j++) {
            double go = dOut[j];
            W1ptr->grad_[j] += go * xptr->value[0];
            xptr->grad_[0] += go * W1ptr->value[j];
            b1ptr->grad_[j] += go;
        }
    };

    Node* h = relu_op(out1);

    Node* out2 = new Node(output_dim);
    double sum = b2.value[0];
    for (std::size_t j = 0; j < hidden_dim; j++) {
        sum += W2.value[j] * h->value[j];
    }
    out2->value[0] = sum;
    out2->parents = {&W2, h, &b2};
    out2->backward_op.backward_func = [out2](const std::vector<double> &dOut){
        Node* W2ptr = out2->parents[0];
        Node* hptr = out2->parents[1];
        Node* b2ptr = out2->parents[2];
        double go = dOut[0];
        for (std::size_t j = 0; j < W2ptr->value.size(); j++) {
            W2ptr->grad_[j] += go * hptr->value[j];
            hptr->grad_[j] += go * W2ptr->value[j];
        }
        b2ptr->grad_[0] += go;
    };

    return out2;
}

int main() {
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (auto &w : W1.value) w = dist(gen);
        for (auto &w : b1.value) w = dist(gen);
        for (auto &w : W2.value) w = dist(gen);
        for (auto &w : b2.value) w = dist(gen);
    }

    std::vector<double> xs;
    std::vector<double> ys;
    for (int i = 0; i < 20; i++) {
        double x_val = (double)i/19 * 2.0 * M_PI;
        xs.push_back(x_val);
        ys.push_back(std::sin(x_val));
    }

    double lr = 0.01;
    int epochs = 2000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        W1.zeroGrad();
        b1.zeroGrad();
        W2.zeroGrad();
        b2.zeroGrad();
        for (int i = 0; i < (int)xs.size(); i++) {
            Node x_node(1);
            x_node.value[0] = xs[i];
            Node* pred = mlp_forward(&x_node);
            Node* loss = mse_loss(pred, ys[i]);
            x_node.zeroGrad();
            pred->zeroGrad();
            loss->zeroGrad();
            backward(loss);
            total_loss += loss->value[0];
            delete pred->parents[1];
            delete pred;
            delete loss;
        }
        for (std::size_t j = 0; j < W1.value.size(); j++) {
            W1.value[j] -= lr * W1.grad_[j];
            b1.value[j] -= lr * b1.grad_[j];
        }
        for (std::size_t j = 0; j < W2.value.size(); j++) {
            W2.value[j] -= lr * W2.grad_[j];
        }
        b2.value[0] -= lr * b2.grad_[0];

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << total_loss / xs.size() << "\n";
        }
    }

    return 0;
}


