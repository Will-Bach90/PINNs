// #include "../include/node.h"
// #include "../include/ops.h"
// #include "../include/autodiff_utils.h"
// #include "../include/linear.h"
// #include "../include/mlp.h"
// #include "../include/utils.h"
#include "../include/neuralnetwork.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>


int main() {
    int total_points = 2000;
    std::vector<double> xs;
    std::vector<double> ys;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_noise(-0.01, 0.01); 

    for(int i = 0; i < total_points; i++) {
        double x_val = (double)i / total_points; // x in [0,1)
        double noise = dist_noise(gen);
        double y_val = 3*std::pow(x_val, 4) - 2*std::pow(x_val, 3) - 2*std::pow(x_val, 2) + x_val*2 + noise;
        xs.push_back(x_val);
        ys.push_back(y_val);
    }

    size_t train_size = 1800;
    size_t test_size = xs.size() - train_size;

    std::vector<double> xs_train(xs.end()-train_size, xs.end());
    std::vector<double> ys_train(ys.end()-train_size, ys.end());
    std::vector<double> xs_test(xs.begin(), xs.end()-train_size);
    std::vector<double> ys_test(ys.begin(), ys.end()-train_size);

    NeuralNetwork nn(
        {1, 10, 10, 1},
        {relu, relu, sigmoid}, // Activations for each layer
        {relu_derivative, relu_derivative, sigmoid_derivative}, // Activation derivatives
        0.001
    );

    std::ofstream out("../../loss.txt");
    size_t batch_size = 32;
    for(int epoch = 0; epoch < 1001; ++epoch) {
        for(size_t i = 0; i < train_size; i+= batch_size) {
            size_t end = std::min(i + batch_size, train_size);
            Tensor inputs(end - i, 1);
            Tensor targets(end - i, 1);

            for(size_t j = i; j < end; ++j) {
                inputs.data_[j-i][0] = xs_train[j];
                targets.data_[j-i][0] = ys_train[j];
            }

            Tensor outputs = nn.forward(inputs);
            nn.backward(targets, 0.01);
        }

        if(epoch % 100 == 0) {
            double loss = 0.0;

            for(size_t i = 0; i < train_size; ++i) {
                Tensor input(1, 1);
                input.data_[0][0] = xs_train[i];
                Tensor target(1, 1);
                target.data_[0][0] = ys_train[i];

                Tensor output = nn.forward(input);
                loss += std::pow(output.data_[0][0]- target.data_[0][0], 2) / 2.0;
            }
            std::cout << "Epoch " << epoch << " Loss: " << loss/train_size << std::endl;
            out << epoch << " " << loss/train_size << "\n";
        }

    }
    out.close();

    std::ofstream pred_file("../../predictions_complex.txt");
    pred_file << "#x true_y pred_y\n";
    for (size_t i = 0; i < test_size; ++i) {
        Tensor inputs(1, 1);
        inputs.data_[0][0] = xs_test[i];

        Tensor outputs = nn.forward(inputs);
        double predicted_value = outputs.data_[0][0];
        double true_value = ys_test[i];

        std::cout << "x=" << xs_test[i] << " true=" << true_value 
                  << " pred=" << predicted_value << "\n";

        pred_file << xs_test[i] << " " << true_value << " " << predicted_value << "\n";

    }

    pred_file.close();




    // double lr = 0.001;
    // double weight_decay = 1e-4;
    // MLP net({1, 10, 1});
    // std::ofstream out("../../loss.txt");
    // for(int epoch = 0; epoch < 5001; epoch++) {
    //     double total_loss = 0.0;

    //     for(auto &layer : net.layers) {
    //         layer.W.zeroGrad();
    //         layer.b.zeroGrad();
    //     }
    //     for ( size_t i = 0; i < xs_train.size(); i++) {
    //         Node x_node(1);
    //         x_node.value[0] = xs_train[i];

    //         Node* pred = net.forward(&x_node);
    //         Node* loss = mse_loss(pred, ys_train[i]);

    //         x_node.zeroGrad();
    //         pred->zeroGrad();
    //         loss->zeroGrad();

    //         backward(loss);

    //         total_loss += loss->value[0]; // 1.94378e-07

    //         delete pred->parents[0];
    //         delete pred;
    //         delete loss;
    //     }
    //     for(auto &layer : net.layers) {
    //         for(size_t j = 0; j < layer.W.value.size(); j++) {
    //             // layer.W.value[j] = layer.W.value[j] - lr*layer.W.grad_[j];
    //             layer.W.value[j] -= lr * (layer.W.grad_[j] + weight_decay * layer.W.value[j]);
    //         }
    //         for(size_t j = 0; j < layer.b.value.size(); j++) {
    //             layer.b.value[j] = layer.b.value[j] - lr*layer.b.grad_[j];
    //         }
    //     }
    //     out << epoch << " " << (total_loss / xs_train.size()) << "\n";
    //     if(epoch % 100 == 0) {
    //         std::cout << "MLP Epoch " << epoch << " Loss: " << total_loss / xs_train.size() << "\n";
    //     }

    // }
    // out.close();

    // std::ofstream pred_file("../../predictions_complex.txt");
    // pred_file << "#x true_y pred_y\n";
    // for (size_t i = 0; i < xs.size(); i++) {
    //     Node x_node(1);
    //     x_node.value[0] = xs[i];

    //     Node* pred = net.forward(&x_node);
    //     double predicted_value = pred->value[0];
    //     double true_value = ys[i];

    //     std::cout << "x=" << xs[i] << " true=" << true_value 
    //               << " pred=" << predicted_value << "\n";

    //     pred_file << xs[i] << " " << true_value << " " << predicted_value << "\n";

    //     delete pred->parents[0];
    //     delete pred;
    // }

    // pred_file.close();

    // std::ofstream train_file("../../training_complex.txt");
    // train_file << "#x true_y pred_y\n";
    // for (size_t i = 0; i < xs_train.size(); i++) {
    //     Node x_node(1);
    //     x_node.value[0] = xs_train[i];

    //     Node* pred = net.forward(&x_node);
    //     double predicted_value = pred->value[0];
    //     double true_value = ys_train[i];

    //     std::cout << "x=" << xs_train[i] << " true=" << true_value 
    //               << " pred=" << predicted_value << "\n";

    //     train_file << xs_train[i] << " " << true_value << " " << predicted_value << "\n";

    //     delete pred->parents[0];
    //     delete pred;
    // }

    // train_file.close();
    return 0;
}