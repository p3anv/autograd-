#include "autograd/autograd_engine.hpp"
#include <stdexcept>

namespace autograd {

void AutogradEngine::run(std::shared_ptr<FunctionNode> root_node, const Tensor& root_grad) {
    // Placeholder implementation - to be fully implemented in Phase 4
    (void)root_node;
    (void)root_grad;
    throw std::runtime_error("AutogradEngine::run not yet implemented");
}

} // namespace autograd