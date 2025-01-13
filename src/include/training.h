#include "neuralnetwork.h"

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

void train_pinn(NeuralNetwork &nn, const std::vector<std::vector<double> > &accelerations,
                const std::vector<std::vector<double> > &positions, const std::vector<double> &times,
                size_t epochs, double learning_rate) {
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