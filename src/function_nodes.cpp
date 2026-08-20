#include "autograd/function_nodes.hpp"
#include "autograd/tensor_impl.hpp"
#include <stdexcept>

namespace autograd {

FunctionNode::FunctionNode(std::vector<std::size_t> output_shape)
    : output_shape_(std::move(output_shape))
{
}

const std::vector<std::size_t>& FunctionNode::output_shape() const noexcept {
    return output_shape_;
}

} // namespace autograd