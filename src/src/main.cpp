#include "../include/node.h"
#include "../include/ops.h"
#include "../include/autodiff_utils.h"
#include <iostream>

int main() {
    Node W(2); W.value = {0.5, -0.3};
    Node x(2); x.value = {1.0, 2.0};
    Node b(1); b.value = {0.1};

    Node* y = linear(&W, &x, &b);
    double target = 0.0;
    Node* loss = mse_loss(y, target);

    W.zeroGrad();
    x.zeroGrad();
    b.zeroGrad();
    y->zeroGrad();
    loss->zeroGrad();

    backward(loss);

    std::cout << "Loss: " << loss->value[0] << "\n";
    std::cout << "dLoss/dW: " << W.grad[0] << ", " << W.grad[1] << "\n";
    std::cout << "dLoss/dx: " << x.grad[0] << ", " << x.grad[1] << "\n";
    std::cout << "dLoss/db: " << b.grad[0] << "\n";

    return 0;
}
