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
}