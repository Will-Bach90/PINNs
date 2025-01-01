// #include "../include/node.h"
// #include "../include/ops.h"
// #include "../include/autodiff_utils.h"
// #include "../include/linear.h"
// #include "../include/mlp.h"
// #include "../include/utils.h"
#include "../include/neuralnetwork.h"
#include "../include/simulated_data.h"
#include "../include/loss.h"
#include "../include/training.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>

void evaluate_model(NeuralNetwork &nn, const std::vector<std::vector<double>> &accelerations,
                    const std::vector<std::vector<double>> &positions, const std::vector<double> &times, const std::string &output_file) {
    std::ofstream file(output_file);
    file << "time,x_true,y_true,z_true,x_pred,y_pred,z_pred\n";

    double mse_x = 0.0, mse_y = 0.0, mse_z = 0.0;

    for (size_t i = 0; i < times.size(); ++i) {
        Tensor input(1, 4);
        input.data_[0][0] = accelerations[i][0];
        input.data_[0][1] = accelerations[i][1];
        input.data_[0][2] = accelerations[i][2];
        input.data_[0][3] = times[i];

        Tensor output = nn.forward(input);

        double x_pred = output.data_[0][0];
        double y_pred = output.data_[0][1];
        double z_pred = output.data_[0][2];

        double x_true = positions[i][0];
        double y_true = positions[i][1];
        double z_true = positions[i][2];

        mse_x += std::pow(x_true - x_pred, 2);
        mse_y += std::pow(y_true - y_pred, 2);
        mse_z += std::pow(z_true - z_pred, 2);

        file << times[i] << "," << x_true << "," << y_true << "," << z_true << ","
             << x_pred << "," << y_pred << "," << z_pred << "\n";
    }

    file.close();

    mse_x /= times.size();
    mse_y /= times.size();
    mse_z /= times.size();

    std::cout << "Mean Squared Error (X): " << mse_x << std::endl;
    std::cout << "Mean Squared Error (Y): " << mse_y << std::endl;
    std::cout << "Mean Squared Error (Z): " << mse_z << std::endl;
}

int main() {

    size_t points = 200;
    double dt = 0.01;
    
    auto [accelerations, positions, times] = simulate_imu_data(points, dt);

    size_t epochs = 10000;
    double learning_rate = 0.01;
    auto optimizer = std::make_shared<SGD>();
    NeuralNetwork nn(
        {4, 20, 6}, // Inputs: 3 accelerations + 1 time, Outputs: 3 positions + 3 velocities
        {sigmoid, sigmoid}, // Activation functions
        {sigmoid_derivative, sigmoid_derivative},
        learning_rate,
        optimizer
    );

    train_pinn(nn, accelerations, positions, times, epochs, learning_rate, dt);


    evaluate_model(nn, accelerations, positions, times, "../../pinn_predictions.csv");

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