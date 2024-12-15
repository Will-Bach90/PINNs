// #include "../include/node.h"
// #include "../include/ops.h"
// #include "../include/autodiff_utils.h"
#include "../include/linear.h"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>


int main() {
    std::size_t input_dim = 1;
    std::size_t hidden_dim = 20;
    std::size_t output_dim = 1;

    Node W1(hidden_dim);
    Node b1(hidden_dim);
    Node W2(output_dim * hidden_dim);
    Node b2(output_dim);

    Linear layer(1,1);
    std::vector<double> xs;
    xs.push_back(0); xs.push_back(1); xs.push_back(2); xs.push_back(3);
    std::vector<double> ys;
    ys.push_back(0); ys.push_back(2); ys.push_back(4); ys.push_back(6);

    double lr = 0.1;
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_loss = 0;
        layer.W.zeroGrad(); layer.b.zeroGrad();
        for (size_t i = 0; i < xs.size(); i++) {

            Node x_node(1); 
            x_node.value[0] = xs[i];

            Node* pred = layer.forward(&x_node);
            Node* loss = mse_loss(pred, ys[i]);

            x_node.zeroGrad(); 
            pred->zeroGrad(); 
            loss->zeroGrad();

            backward(loss);
            total_loss += loss->value[0];
            delete pred; 
            delete loss;
        }

        for (size_t i = 0; i < layer.W.value.size(); i++)
            layer.W.value[i] -= lr * layer.W.grad_[i];
        for (size_t i = 0; i < layer.b.value.size(); i++)
            layer.b.value[i] -= lr * layer.b.grad_[i];

        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << " Loss: " << total_loss / xs.size() << "\n";
    }
    return 0;
}