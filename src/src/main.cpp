// #include "../include/node.h"
// #include "../include/ops.h"
// #include "../include/autodiff_utils.h"
// #include "../include/linear.h"
// #include "../include/mlp.h"
// #include "../include/utils.h"
#include "../include/neuralnetwork.h"
#include "../include/loss.h"
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

void train_pinn(NeuralNetwork &nn, const std::vector<std::vector<double>> &accelerations,
                const std::vector<std::vector<double>> &positions, const std::vector<double> &times,
                size_t epochs, double learning_rate, double dt) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double data_loss = 0.0;
        double physics_loss = 0.0;

        for (size_t i = 0; i < times.size(); ++i) {
            Tensor input(1, 4); // Inputs: (ax, ay, az, t)
            input.data_[0][0] = accelerations[i][0];
            input.data_[0][1] = accelerations[i][1];
            input.data_[0][2] = accelerations[i][2];
            input.data_[0][3] = times[i];

            Tensor target(1, 3); // Outputs: (x, y, z)
            target.data_[0][0] = positions[i][0];
            target.data_[0][1] = positions[i][1];
            target.data_[0][2] = positions[i][2];

            Tensor output = nn.forward(input);
            nn.backward(target, learning_rate);

            for (size_t j = 0; j < 3; ++j) {
                data_loss += std::pow(output.data_[0][j] - target.data_[0][j], 2) / 2.0;
            }
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch
                      << " Data Loss: " << data_loss / times.size() << std::endl;
        }
    }
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>> simulate_imu_data(size_t points, double dt) {
    std::vector<std::vector<double>> accelerations(points, std::vector<double>(3));
    std::vector<std::vector<double>> positions(points, std::vector<double>(3));
    std::vector<double> times(points);

    double x = 0, y = 0, z = 0;
    double vx = 0, vy = 0, vz = 0;

    for (size_t i = 0; i < points; ++i) {
        double ax = random_double(-0.1, 0.1); // Simulated acceleration with noise
        double ay = random_double(-0.1, 0.1);
        double az = random_double(-0.1, 0.1);

        vx += ax * dt;
        vy += ay * dt;
        vz += az * dt;

        x += vx * dt;
        y += vy * dt;
        z += vz * dt;

        accelerations[i] = {ax, ay, az};
        positions[i] = {x, y, z};
        times[i] = i * dt;
    }

    return {accelerations, positions, times};
}

int main() {

    size_t points = 1000;
    double dt = 0.01;

    // Simulate IMU data
    auto [accelerations, positions, times] = simulate_imu_data(points, dt);

    size_t epochs = 1000;
    double learning_rate = 0.01;
    auto optimizer = std::make_shared<SGD>();
    NeuralNetwork nn(
        {4, 20, 20, 6}, // Inputs: 3 accelerations + 1 time, Outputs: 3 positions + 3 velocities
        {relu, relu, relu}, // Activation functions
        {relu_derivative, relu_derivative, relu_derivative},
        learning_rate,
        optimizer
    );

    train_pinn(nn, accelerations, positions, times, epochs, learning_rate, dt);


    // size_t total_points = 500;
    // std::vector<double> xs;
    // std::vector<double> ys;

    // std::mt19937 gen(std::random_device{}());
    // std::uniform_real_distribution<double> dist_noise(-0.01, 0.01);

    // for (size_t i = 0; i < total_points; i++) {
    //     double x_val = static_cast<double>(i) / total_points;
    //     double noise = dist_noise(gen);
    //     double y_val = 3 * std::pow(x_val, 4) - 2 * std::pow(x_val, 3) - 2 * std::pow(x_val, 2) + x_val * 2 + noise;
    //     xs.push_back(x_val);
    //     ys.push_back(y_val);
    // }

    // // Normalize data
    // double max_x = *std::max_element(xs.begin(), xs.end());
    // double min_x = *std::min_element(xs.begin(), xs.end());

    // double max_y = *std::max_element(ys.begin(), ys.end());
    // double min_y = *std::min_element(ys.begin(), ys.end());

    // std::vector<double> xs_normalized(xs.size());
    // std::vector<double> ys_normalized(ys.size());

    // for (size_t i = 0; i < xs.size(); ++i) {
    //     xs_normalized[i] = (xs[i] - min_x) / (max_x - min_x); // Scale to [0, 1]
    //     ys_normalized[i] = (ys[i] - min_y) / (max_y - min_y); 
    // }

    // size_t train_size = 300;
    // size_t epochs = 10001;
    // double learning_rate = 0.01;

    // std::vector<double> xs_train(xs_normalized.begin(), xs_normalized.begin() + train_size);
    // std::vector<double> ys_train(ys_normalized.begin(), ys_normalized.begin() + train_size);

    // std::vector<double> xs_test(xs_normalized.begin() + train_size, xs_normalized.end());
    // std::vector<double> ys_test(ys_normalized.begin() + train_size, ys_normalized.end());

    // auto optimizer = std::make_shared<SGD>();

    // NeuralNetwork nn(
    //     {1, 10, 10, 1},
    //     {sigmoid, sigmoid, sigmoid},
    //     {sigmoid_derivative, sigmoid_derivative, sigmoid_derivative},
    //     learning_rate,
    //     optimizer
    //     );

    // train(nn, xs, ys, epochs, learning_rate);

    // std::ofstream pred_file("../../predictions_complex.txt");
    // pred_file << "#x true_y pred_y\n";
    // for (size_t i = 0; i < xs_test.size(); ++i) {
    //     Tensor inputs(1, 1);
    //     inputs.data_[0][0] = xs_test[i];

    //     Tensor outputs = nn.forward(inputs);
    //     double predicted_value = outputs.data_[0][0];
    //     double true_value = ys_test[i];

    //     pred_file << xs_test[i] << " " << true_value << " " << predicted_value << "\n";

    // }

    // pred_file.close();

    // std::ofstream train_file("../../training_complex.txt");
    // pred_file << "#x true_y pred_y\n";
    // for (size_t i = 0; i < xs.size(); ++i) {
    //     Tensor inputs(1, 1);
    //     inputs.data_[0][0] = xs[i];

    //     Tensor outputs = nn.forward(inputs);
    //     double predicted_value = outputs.data_[0][0];
    //     double true_value = ys[i];

    //     train_file << xs[i] << " " << true_value << " " << predicted_value << "\n";

    // }

    // train_file.close();
    // return 0;
}