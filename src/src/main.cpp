// #include "../include/node.h"
// #include "../include/ops.h"
// #include "../include/autodiff_utils.h"
#include "../include/linear.h"
#include "../include/mlp.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>


int main() {
    int total_points = 1000;
    std::vector<double> xs;
    std::vector<double> ys;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_noise(-0.1, 0.1); // noise range

    for(int i = 0; i < total_points; i++) {
        double x_val = (double)i / total_points; // x in [0,1)
        double noise = dist_noise(gen);
        double y_val = std::sin(2.0 * M_PI * x_val) + noise; 
        xs.push_back(x_val);
        ys.push_back(y_val);
    }

    size_t train_size = 900;
    size_t test_size = xs.size() - train_size;

    std::vector<double> xs_train(xs.begin(), xs.begin() + train_size);
    std::vector<double> ys_train(ys.begin(), ys.begin() + train_size);
    std::vector<double> xs_test(xs.begin() + train_size, xs.end());
    std::vector<double> ys_test(ys.begin() + train_size, ys.end());

    double lr = 0.01;
    MLP net({1, 20, 1});
    std::ofstream out("../../loss.txt");
    for(int epoch = 0; epoch < 1001; epoch++) {
        double total_loss = 0.0;

        for(auto &layer : net.layers) {
            layer.W.zeroGrad();
            layer.b.zeroGrad();
        }
        for ( size_t i = 0; i < xs_train.size(); i++) {
            Node x_node(1);
            x_node.value[0] = xs_train[i];

            Node* pred = net.forward(&x_node);
            Node* loss = mse_loss(pred, ys_train[i]);

            x_node.zeroGrad();
            pred->zeroGrad();
            loss->zeroGrad();

            backward(loss);

            total_loss += loss->value[0]; // 1.94378e-07

            // delete pred->parents[0];
            delete pred;
            delete loss;
        }
        for(auto &layer : net.layers) {
            for(size_t j = 0; j < layer.W.value.size(); j++) {
                layer.W.value[j] = layer.W.value[j] - lr*layer.W.grad_[j];
            }
            for(size_t j = 0; j < layer.b.value.size(); j++) {
                layer.b.value[j] = layer.b.value[j] - lr*layer.b.grad_[j];
            }
        }
        out << epoch << " " << (total_loss / xs_train.size()) << "\n";
        if(epoch % 100 == 0) {
            std::cout << "MLP Epoch " << epoch << " Loss: " << total_loss / xs_train.size() << "\n";
        }

    }
    out.close();

    std::ofstream pred_file("../../predictions_complex.txt");
    pred_file << "#x true_y pred_y\n";
    for (size_t i = 0; i < xs_test.size(); i++) {
        Node x_node(1);
        x_node.value[0] = xs_test[i];

        Node* pred = net.forward(&x_node);
        double predicted_value = pred->value[0];
        double true_value = ys_test[i];

        std::cout << "x=" << xs_test[i] << " true=" << true_value 
                  << " pred=" << predicted_value << "\n";

        pred_file << xs_test[i] << " " << true_value << " " << predicted_value << "\n";

        delete pred->parents[0];
        delete pred;
    }

    pred_file.close();
    return 0;
}