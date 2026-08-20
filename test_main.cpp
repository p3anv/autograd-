#include <autograd/tensor.hpp>
#include <iostream>

int main() {
    autograd::Tensor t = autograd::Tensor::zeros({2, 3}, false);
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < t.shape().size(); ++i) {
        std::cout << t.shape()[i];
        if (i < t.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Test setting and getting a value
    if (t.numel() > 0) {
        t({0, 0}) = 5.0f;
        std::cout << "Value at (0,0): " << t({0, 0}) << std::endl;
    }

    return 0;
}