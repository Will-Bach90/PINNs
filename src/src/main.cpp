#include "../include/node.h"
#include "../include/ops.h"
#include "../include/autodiff_utils.h"
#include <iostream>

int main() {
    Node W(2); 
    W.value = {0.5, -0.3}; 

    std::vector<int> r;
    Node x(2); 
    x.value = {1.0, 2.0};  

    Node b(1);    
    b.value = {0.1};

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
    std::cout << "dLoss/dW: " << W.grad_[0] << ", " << W.grad_[1] << "\n";
    std::cout << "dLoss/dx: " << x.grad_[0] << ", " << x.grad_[1] << "\n";
    std::cout << "dLoss/db: " << b.grad_[0] << "\n";

    return 0;
}
