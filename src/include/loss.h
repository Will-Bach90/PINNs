#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include "neuralnetwork.h"

double compute_physics_loss(NeuralNetwork &nn, const std::vector<double> &times, const std::vector<std::vector<double>> &accelerations, double dt) {
    
    double physics_loss = 0.0;

    for (size_t i = 0; i < times.size(); ++i) {
        Tensor input(1, 4); // Inputs: (ax, ay, az, t)
        input.data_[0][0] = accelerations[i][0]; // ax
        input.data_[0][1] = accelerations[i][1]; // ay
        input.data_[0][2] = accelerations[i][2]; // az
        input.data_[0][3] = times[i]; // t

        Tensor output = nn.forward(input);

        // Outputs: [x, y, z, vx, vy, vz]
        double vx = output.data_[0][3];
        double vy = output.data_[0][4];
        double vz = output.data_[0][5];

        double ax = accelerations[i][0];
        double ay = accelerations[i][1];
        double az = accelerations[i][2];

        // Approximate dv/dt using IMU accelerations and PINN corrections
        double dvx_dt = vx - ax;
        double dvy_dt = vy - ay;
        double dvz_dt = vz - az;

        physics_loss += std::pow(dvx_dt, 2) + std::pow(dvy_dt, 2) + std::pow(dvz_dt, 2);
    }

    return physics_loss / times.size();
}


#endif