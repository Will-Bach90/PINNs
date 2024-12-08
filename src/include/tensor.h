#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>

class Tensor {
public:
    Tensor() = default;
    Tensor(std::size_t size) : data_(size, 0.0) {}

    double& operator[](std::size_t idx) { return data_[idx]; }
    const double& operator[](std::size_t idx) const { return data_[idx]; }

    std::size_t size() const { return data_.size(); }

    void resize(std::size_t size) { data_.resize(size, 0.0); }

private:
    std::vector<double> data_;
};

#endif 