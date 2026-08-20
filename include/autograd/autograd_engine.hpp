#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "autograd/tensor.hpp"
#include "autograd/function_nodes.hpp"

namespace autograd {

class AutogradEngine {
public:
    void run(std::shared_ptr<FunctionNode> root_node, const Tensor& root_grad);

private:
    // Temporary gradient map for one backward pass
    std::unordered_map<FunctionNode*, Tensor> node_grads_;
};

} // namespace autograd