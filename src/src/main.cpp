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
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

std::vector<std::vector<double > > load_csv(const std::string &filename) {
    std::vector<std::vector<double > > data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value)); // Convert string to double
            } catch (const std::invalid_argument &) {
                continue;
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>>
preprocess_data(const std::vector<std::vector<double>> &imu_data,
                const std::vector<std::vector<double>> &gps_data) {
    std::vector<std::vector<double>> aligned_accel;
    std::vector<std::vector<double>> aligned_positions;
    std::vector<double> aligned_times;

    size_t imu_idx = 0, gps_idx = 0;

    while (imu_idx < imu_data.size() && gps_idx < gps_data.size()) {
        double imu_time = imu_data[imu_idx][0];
        double gps_time = gps_data[gps_idx][0];

        if (std::abs(imu_time - gps_time) < 0.01) {
            aligned_times.push_back(imu_time);
            aligned_accel.push_back({imu_data[imu_idx][4], imu_data[imu_idx][5], imu_data[imu_idx][6]});
            aligned_positions.push_back({gps_data[gps_idx][2], gps_data[gps_idx][3], gps_data[gps_idx][4]});
            ++imu_idx;
            ++gps_idx;
        } else if (imu_time < gps_time) {
            ++imu_idx; // IMU ahead, move to next IMU entry
        } else {
            ++gps_idx; // GPS ahead, move to next GPS entry
        }
    }

    return {aligned_accel, aligned_positions, aligned_times};
}

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

    auto imu_data = load_csv("../../onboardposition.csv");  // Function to parse IMU data
    auto gps_data = load_csv("../../onboardgps.csv");

    auto [aligned_accel, aligned_positions, aligned_times] = preprocess_data(imu_data, gps_data);

    aligned_accel = normalize_2d(aligned_accel);
    aligned_positions = normalize_2d(aligned_positions);
    aligned_times = normalize(aligned_times);

    double learning_rate = 0.01;

    auto optimizer = std::make_shared<SGD>();
    NeuralNetwork nn(
        {4, 20, 20, 3}, // Input: (ax, ay, az, t), Output: (x, y, z)
        {sigmoid, sigmoid, sigmoid},
        {sigmoid_derivative, sigmoid_derivative, sigmoid_derivative},
        learning_rate,
        optimizer
        );

    size_t epochs = 8000;
    double dt = 0.01;

    train_pinn(nn, aligned_accel, aligned_positions, aligned_times, epochs, learning_rate, dt);

    std::ofstream outfile("../../pinn_predictions.csv");
    outfile << "time,x_true,y_true,z_true,x_pred,y_pred,z_pred\n";
    for (size_t i = 0; i < aligned_times.size(); ++i) {
        Tensor input(1, 4);
        input.data_[0][0] = aligned_accel[i][0];
        input.data_[0][1] = aligned_accel[i][1];
        input.data_[0][2] = aligned_accel[i][2];
        input.data_[0][3] = aligned_times[i];

        Tensor output = nn.forward(input);
        outfile << aligned_times[i] << ","
                << aligned_positions[i][0] << "," << aligned_positions[i][1] << "," << aligned_positions[i][2] << ","
                << output.data_[0][0] << "," << output.data_[0][1] << "," << output.data_[0][2] << "\n";
    }
    outfile.close();

    // size_t points = 200;
    // double dt = 0.01;
    
    // auto [accelerations, positions, times] = simulate_imu_data(points, dt);

    // size_t epochs = 10000;
    // double learning_rate = 0.01;
    // auto optimizer = std::make_shared<SGD>();
    // NeuralNetwork nn(
    //     {4, 20, 6}, // Inputs: 3 accelerations + 1 time, Outputs: 3 positions + 3 velocities
    //     {sigmoid, sigmoid}, // Activation functions
    //     {sigmoid_derivative, sigmoid_derivative},
    //     learning_rate,
    //     optimizer
    // );

    // train_pinn(nn, accelerations, positions, times, epochs, learning_rate, dt);


    // evaluate_model(nn, accelerations, positions, times, "../../pinn_predictions.csv");
}