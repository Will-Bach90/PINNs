#ifndef SIMULATED_DATA_H
#define SIMULATED_DATA_H

#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <tuple>
#include "denselayer.h"

std::tuple<std::vector<std::vector<double> >, std::vector<std::vector<double> >, std::vector<double> > simulate_imu_data(size_t points, double dt) {
    std::vector<std::vector<double > > accelerations(points, std::vector<double>(3));
    std::vector<std::vector<double> > positions(points, std::vector<double>(3));
    std::vector<double> times(points);

    double x = 0, y = 0, z = 0;
    double vx = 0, vy = 0, vz = 0;

    for (size_t i = 0; i < points; ++i) {
        double ax = random_double(-0.001, 0.001); // Simulated acceleration with noise
        double ay = random_double(-0.001, 0.001);
        double az = random_double(-0.001, 0.001);

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

    accelerations = normalize_2d(accelerations);
    positions = normalize_2d(positions);
    times = normalize(times);

    return {accelerations, positions, times};
}

#endif