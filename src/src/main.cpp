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

void train(NeuralNetwork &nn, const std::vector<double> &xs_train, const std::vector<double> &ys_train, size_t epochs, double learning_rate) {
    std::ofstream out("../../loss.txt");
    size_t train_size = xs_train.size();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < train_size; ++i) {
            Tensor input(1, 1);
            input.data_[0][0] = xs_train[i];

            Tensor target(1, 1);
            target.data_[0][0] = ys_train[i];

            Tensor output = nn.forward(input);
            nn.backward(target, learning_rate);

            loss += std::pow(output.data_[0][0] - target.data_[0][0], 2) / 2.0;
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss / train_size << std::endl;
            out << epoch << " " << loss/train_size << "\n";
        }
    }
    out.close();
}

int main() {
    size_t total_points = 500;
    std::vector<double> xs;
    std::vector<double> ys;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_noise(-0.01, 0.01);

    for (size_t i = 0; i < total_points; i++) {
        double x_val = static_cast<double>(i) / total_points;
        double noise = dist_noise(gen);
        double y_val = 3 * std::pow(x_val, 4) - 2 * std::pow(x_val, 3) - 2 * std::pow(x_val, 2) + x_val * 2 + noise;
        xs.push_back(x_val);
        ys.push_back(y_val);
    }

    // Normalize data
    double max_x = *std::max_element(xs.begin(), xs.end());
    double min_x = *std::min_element(xs.begin(), xs.end());

    double max_y = *std::max_element(ys.begin(), ys.end());
    double min_y = *std::min_element(ys.begin(), ys.end());

    std::vector<double> xs_normalized(xs.size());
    std::vector<double> ys_normalized(ys.size());

    for (size_t i = 0; i < xs.size(); ++i) {
        xs_normalized[i] = (xs[i] - min_x) / (max_x - min_x); // Scale to [0, 1]
        ys_normalized[i] = (ys[i] - min_y) / (max_y - min_y); // Scale to [0, 1]
    }

    size_t train_size = 300;
    size_t epochs = 10001;
    double learning_rate = 0.1;

    std::vector<double> xs_train(xs_normalized.begin(), xs_normalized.begin() + train_size);
    std::vector<double> ys_train(ys_normalized.begin(), ys_normalized.begin() + train_size);

    std::vector<double> xs_test(xs_normalized.begin() + train_size, xs_normalized.end());
    std::vector<double> ys_test(ys_normalized.begin() + train_size, ys_normalized.end());

    NeuralNetwork nn(
        {1, 10, 10, 1},
        {sigmoid, sigmoid, sigmoid},
        {sigmoid_derivative, sigmoid_derivative, sigmoid_derivative},
        learning_rate);

    train(nn, xs, ys, epochs, learning_rate);

    std::ofstream pred_file("../../predictions_complex.txt");
    pred_file << "#x true_y pred_y\n";
    for (size_t i = 0; i < xs_test.size(); ++i) {
        Tensor inputs(1, 1);
        inputs.data_[0][0] = xs_test[i];

        Tensor outputs = nn.forward(inputs);
        double predicted_value = outputs.data_[0][0];
        double true_value = ys_test[i];

        pred_file << xs_test[i] << " " << true_value << " " << predicted_value << "\n";

    }

    pred_file.close();

    std::ofstream train_file("../../training_complex.txt");
    pred_file << "#x true_y pred_y\n";
    for (size_t i = 0; i < xs.size(); ++i) {
        Tensor inputs(1, 1);
        inputs.data_[0][0] = xs[i];

        Tensor outputs = nn.forward(inputs);
        double predicted_value = outputs.data_[0][0];
        double true_value = ys[i];

        train_file << xs[i] << " " << true_value << " " << predicted_value << "\n";

    }

    train_file.close();
    return 0;
}