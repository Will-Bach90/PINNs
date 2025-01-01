#include "../include/tensor.h"

Tensor::Tensor(size_t rows, size_t cols) 
    : data_(rows, std::vector<double>(cols, 0.0)) 
    {}

size_t Tensor::rows() const { 
    return data_.size(); 
}
size_t Tensor::cols() const { 
    return data_[0].size(); 
}

Tensor Tensor::operator*(const Tensor &other) const {
    if(cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    Tensor result(rows(), other.cols());
    for(size_t i = 0; i < rows(); ++i) {
        for(size_t j = 0; j < other.cols(); ++j) {
            for(size_t k = 0; k < cols(); ++k) {
                result.data_[i][j] += data_[i][k] * other.data_[k][j];
            }
        }
    }
    return result;
}

Tensor Tensor::operator+(const Tensor &other) const {
    if(rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }

    Tensor result(rows(), cols());

    for(size_t i = 0; i < rows(); ++i) {
        for(size_t j = 0; j < cols(); ++j) {
            result.data_[i][j] = data_[i][j] + other.data_[i][j];
        }
    }

    return result;
}

Tensor Tensor::apply(const std::function<double (double) > &func) const {
    Tensor result(rows(), cols());
    for(size_t i = 0; i < rows(); ++i) {
        for(size_t j = 0; j < cols(); ++j) {
            result.data_[i][j]  = func(data_[i][j]);
        }
    }
    return result;
}
