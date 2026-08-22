#include "autograd/autograd_engine.hpp"
#include <stdexcept>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <numeric>

namespace autograd {

void AutogradEngine::run(std::shared_ptr<FunctionNode> root_node, const Tensor& root_grad) {
    if (!root_node) {
        throw std::invalid_argument("root_node cannot be null");
    }

    // Clear gradient map for this run
    node_grads_.clear();

    // Step 1: Topological sort - collect all nodes in the computation graph
    std::vector<std::shared_ptr<FunctionNode>> nodes;
    std::unordered_set<const FunctionNode*> visited;

    // Lambda for DFS traversal
    std::function<void(const std::shared_ptr<FunctionNode>&)> dfs =
        [&](const std::shared_ptr<FunctionNode>& node) {
            if (!node || visited.count(node.get())) {
                return;
            }

            visited.insert(node.get());

            // Visit parents first - only intermediate parents (nodes) need to be visited
            for (const auto& parent_edge : node->parents_) {
                if (parent_edge.intermediate_parent()) {
                    dfs(parent_edge.intermediate_parent());
                }
            }

            // Add current node after visiting parents (post-order)
            nodes.push_back(node);
        };

    // Start DFS from root
    dfs(root_node);

    // Step 2: Initialize root gradient
    node_grads_[root_node.get()] = root_grad;

    // Step 3: Process nodes in reverse topological order (from output to input)
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        auto node = *it;
        auto node_iter = node_grads_.find(node.get());

        // Skip if no gradient for this node (shouldn't happen for nodes in graph)
        if (node_iter == node_grads_.end()) {
            continue;
        }

        Tensor grad_output = node_iter->second;
        std::vector<Tensor> grad_inputs = node->backward({grad_output});

        // Step 4: Distribute gradients to parents
        size_t parent_idx = 0;
        for (const auto& parent_edge : node->parents_) {
            if (parent_idx >= grad_inputs.size()) {
                throw std::runtime_error("Node backward returned insufficient gradients");
            }

            const Tensor& grad_input = grad_inputs[parent_idx];

            // Accumulate gradient to parent
            if (parent_edge.is_leaf()) {
                // Parent is a leaf tensor - accumulate gradient directly
                auto leaf_parent = parent_edge.leaf_parent();
                if (leaf_parent && leaf_parent->requires_grad_) {
                    leaf_parent->accumulate_grad(grad_input);
                }
            } else if (parent_edge.is_intermediate()) {
                // Parent is an intermediate node - accumulate gradient for later processing
                auto parent_node = parent_edge.intermediate_parent();
                if (parent_node) {
                    // Get or create gradient tensor for parent node
                    Tensor& parent_grad = node_grads_[parent_node.get()];

                    // Initialize gradient tensor with zeros if not yet set or empty
                    if (parent_grad.numel() == 0) {
                        parent_grad = Tensor::zeros(parent_node->output_shape(), false);
                    }

                    // Add gradient to parent's accumulated gradient
                    // Both tensors should have the same shape at this point
                    const float* grad_data = grad_input.data();
                    float* parent_grad_data = parent_grad.data();

                    for (std::size_t i = 0; i < parent_grad.numel(); ++i) {
                        parent_grad_data[i] += grad_data[i];
                    }
                }
            }

            parent_idx++;
        }

        // Clear gradient for this node to free memory (optional)
        node_grads_.erase(node_iter);
    }
}

} // namespace autograd