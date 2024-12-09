#ifndef OPS_H
#define OPS_H

#include "node.h"
#include <cassert>


inline Node* linear(Node* W, Node* x, Node* b) {
    assert(W->value.size() == x->value.size());
    assert(b->value.size() == 1);

    Node* out = new Node(1);
    double sum = b->value[0];
    for (std::size_t i = 0; i < x->value.size(); i++) {
        sum += W->value[i] * x->value[i];
    }
    out->value[0] = sum;

    out->parents = {W, x, b};

    out->backward_op.backward_func = [out,W,x,b](const std::vector<double> &dOut) {
        double go = dOut[0];
        for (std::size_t i = 0; i < W->value.size(); i++) {
            W->grad_[i] += go * x->value[i];
        }
        for (std::size_t i = 0; i < x->value.size(); i++) {
            x->grad_[i] += go * W->value[i];
        }
        b->grad_[0] += go;
    };

    return out;
}

inline Node* matmul(Node* W, Node* x, std::size_t output_dim, std::size_t input_dim) {
    Node* out = new Node(output_dim);
    for(std::size_t i = 0; i < output_dim; i++) {
        double sum = 0.0;
        for(std::size_t j = 0; j < input_dim; i++) {
            sum += W->value[i * input_dim + j] * x->value[j];
        }
        out->value[i] = sum;
    }

    out->parents = {W, x};
    out->backward_op.backward_func = [out, W, x, output_dim, input_dim](const std::vector<double>& dOut) {
        for(std::size_t i = 0; i < output_dim; i++) {
            double go = dOut[i];
            for(std::size_t j = 0; j < input_dim; j++) {
                W->grad_[i * input_dim + j] += go * x->value[j];
                x->grad_[j] += go * W->value[i * input_dim + j];
            }
        }
    };

    return out;
}

inline Node* add(Node* y, Node* b, std::size_t output_dim) {
    Node* out = new Node(output_dim);
    double sum = 0.0;
    for (std::size_t i = 0; i < output_dim; i++) {
        out->value[i] = y->value[i] + b->value[i];
    }
    out->parents = {y, b};

    out->backward_op.backward_func = [out, y, b, output_dim](const std::vector<double>& dOut) {
        for (std::size_t i = 0; i < output_dim; i++) {
            y->grad[i] += dOut[i];
            b->grad[i] += dOut[i];
        }
    };
    return out;
}

Node* linear_layer(Node* W, Node* b, Node* x, std::size_t output_dim, std::size_t input_dim) {
    Node* y = matmul(W, x, output_dim, input_dim);
    Node* out = add(y, b, output_dim);
    return out;
}


inline Node* tanh_op(Node* input) {
    Node* out = new Node(input->value.size());
    for (std::size_t i = 0; i < input->value.size(); i++) {
        double val = input->value[i];
        double t = std::tanh(val);
        out->value[i] = t;
    }

    out->parents = {input};

    out->backward_op.backward_func = [out, input](const std::vector<double> &dOut) {
        for (std::size_t i = 0; i < out->value.size(); i++) {
            double t = out->value[i]; 
            double dt = (1.0 - t * t) * dOut[i]; 
            input->grad_[i] += dt;
        }
    };
    return out;
}

inline Node* relu_op(Node* input) {
    Node* out = new Node(input->value.size());
    for (std::size_t i = 0; i < input->value.size(); i++) {
        double val = input->value[i];
        out->value[i] = (val > 0.0) ? val : 0.0;
    }

    out->parents = {input};

    out->backward_op.backward_func = [out,input](const std::vector<double> &dOut) {
        for (std::size_t i = 0; i < out->value.size(); i++) {
            double x_in = input->value[i];
            input->grad_[i] += (x_in > 0.0) ? dOut[i] : 0.0;
        }
    };

    return out;
}

inline Node* sigmoid_op(Node* input) {
    Node* out = new Node(input->value.size());
    for (std::size_t i = 0; i < input->value.size(); i++) {
        double val = input->value[i];
        double s = 1.0 / (1.0 + std::exp(-val));
        out->value[i] = s;
    }

    out->parents = {input};

    out->backward_op.backward_func = [out,input](const std::vector<double> &dOut) {
        for (std::size_t i = 0; i < out->value.size(); i++) {
            double s = out->value[i];
            double ds = s * (1.0 - s) * dOut[i];
            input->grad_[i] += ds;
        }
    };

    return out;
}

inline Node* mse_loss(Node* pred, double target) {
    Node* out = new Node(1);
    double diff = pred->value[0] - target;
    out->value[0] = 0.5 * diff * diff;

    out->parents = {pred};

    out->backward_op.backward_func = [out,pred,target](const std::vector<double> &dOut) {
        double go = dOut[0];
        double diff = pred->value[0] - target;
        pred->grad_[0] += go * diff; 
    };

    return out;
}



inline Node* mlp_forward(Node* x, std::size_t hidden_dim, std::size_t output_dim, Node* W1, Node* b1, Node* W2, Node* b2) {
    
    Node* out1 = new Node(hidden_dim);
    for (std::size_t j = 0; j < hidden_dim; j++) {
        out1->value[j] = W1->value[j] * x->value[0] + b1->value[j];
    }
    out1->parents = {W1, x, b1};
    out1->backward_op.backward_func = [out1, W1, x, b1](const std::vector<double> &dOut){
        for (std::size_t j = 0; j < out1->value.size(); j++) {
            double go = dOut[j];
            W1->grad_[j] += go * x->value[0];
            x->grad_[0] += go * W1->value[j];
            b1->grad_[j] += go;
        }
    };

    Node* h = relu_op(out1);
    Node* out2 = new Node(output_dim);
    double sum = b2->value[0];
    for (std::size_t j = 0; j < hidden_dim; j++) {
        sum += W2->value[j] * h->value[j];
    }
    out2->value[0] = sum;

    out2->parents = {W2, h, b2};
    out2->backward_op.backward_func = [out2, W2, h, b2](const std::vector<double> &dOut){
        double go = dOut[0];
        for (std::size_t j = 0; j < W2->value.size(); j++) {
            W2->grad_[j] += go * h->value[j];
            h->grad_[j] += go * W2->value[j];
        }
        b2->grad_[0] += go;
    };

    return out2;
}


#endif