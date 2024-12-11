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
            Node* pred = mlp_forward(&x_node, hidden_dim, output_dim, &W1, &b1, &W2, &b2);
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