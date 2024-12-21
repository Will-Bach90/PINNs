#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>
#include <cstddef>
#include <cmath>
#include <random>
#include <fstream>

class Tensor {
public:

    std::vector<std::vector<double> > data_;
    Tensor() = default;
    Tensor(size_t rows, size_t cols);

    size_t rows() const;
    size_t cols() const;

    Tensor operator*(const Tensor &) const;

    Tensor operator+(const Tensor &) const;

    Tensor apply(const std::function<double (double) > &) const;

    std::size_t size() const { return data_.size(); }

};

#endif 