#include "autograd/operations.hpp"
#include "autograd/tensor.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace autograd {

Tensor add(const Tensor& a, const Tensor& b) {
    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();
    std::vector<std::size_t> b_shape = b.shape();

    // Handle broadcasting
    std::vector<std::size_t> result_shape;
    std::size_t max_dim = std::max(a_shape.size(), b_shape.size());

    // Pad shapes with 1s on the left (prepend dimensions)
    while (a_shape.size() < max_dim) {
        a_shape.insert(a_shape.begin(), 1);
    }
    while (b_shape.size() < max_dim) {
        b_shape.insert(b_shape.begin(), 1);
    }

    // Compute result shape and check for broadcasting compatibility
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_shape[i] == b_shape[i]) {
            result_shape.push_back(a_shape[i]);
        } else if (a_shape[i] == 1) {
            result_shape.push_back(b_shape[i]);
        } else if (b_shape[i] == 1) {
            result_shape.push_back(a_shape[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();
    const float* b_data = b.data();

    // Handle special cases for efficiency
    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (a_is_scalar && b_is_scalar) {
        // Both scalars
        *result_data = *a_data + *b_data;
    } else if (a_is_scalar) {
        // a is scalar, b is tensor
        float a_val = *a_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_val + b_data[i];
        }
    } else if (b_is_scalar) {
        // b is scalar, a is tensor
        float b_val = *b_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_data[i] + b_val;
        }
    } else {
        // Both are tensors - need to handle broadcasting properly
        // We'll iterate through the result tensor and compute indices for a and b
        std::vector<std::size_t> a_indices(a_shape.size(), 0);
        std::vector<std::size_t> b_indices(b_shape.size(), 0);
        std::vector<std::size_t> result_indices(result_shape.size(), 0);

        std::size_t linear_index = 0;
        while (linear_index < result.numel()) {
            // Compute a index (with broadcasting)
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                if (a_shape[i] == 1) {
                    a_indices[i] = 0;
                } else {
                    a_indices[i] = result_indices[i];
                }
            }

            // Compute b index (with broadcasting)
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                if (b_shape[i] == 1) {
                    b_indices[i] = 0;
                } else {
                    b_indices[i] = result_indices[i];
                }
            }

            // Compute offsets
            std::size_t a_offset = 0;
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                a_offset += a_indices[i];
                if (i < a_shape.size() - 1) {
                    a_offset *= a_shape[i + 1];
                }
            }

            std::size_t b_offset = 0;
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                b_offset += b_indices[i];
                if (i < b_shape.size() - 1) {
                    b_offset *= b_shape[i + 1];
                }
            }

            // Perform operation
            result_data[linear_index] = a_data[a_offset] + b_data[b_offset];

            // Increment result indices (like an odometer)
            for (std::size_t i = result_shape.size(); i > 0; --i) {
                std::size_t idx = i - 1;
                if (++result_indices[idx] < result_shape[idx]) {
                    break;
                }
                result_indices[idx] = 0;
            }

            ++linear_index;
        }
    }

    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();
    std::vector<std::size_t> b_shape = b.shape();

    // Handle broadcasting
    std::vector<std::size_t> result_shape;
    std::size_t max_dim = std::max(a_shape.size(), b_shape.size());

    // Pad shapes with 1s on the left (prepend dimensions)
    while (a_shape.size() < max_dim) {
        a_shape.insert(a_shape.begin(), 1);
    }
    while (b_shape.size() < max_dim) {
        b_shape.insert(b_shape.begin(), 1);
    }

    // Compute result shape and check for broadcasting compatibility
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_shape[i] == b_shape[i]) {
            result_shape.push_back(a_shape[i]);
        } else if (a_shape[i] == 1) {
            result_shape.push_back(b_shape[i]);
        } else if (b_shape[i] == 1) {
            result_shape.push_back(a_shape[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();
    const float* b_data = b.data();

    // Handle special cases for efficiency
    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (a_is_scalar && b_is_scalar) {
        // Both scalars
        *result_data = *a_data - *b_data;
    } else if (a_is_scalar) {
        // a is scalar, b is tensor
        float a_val = *a_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_val - b_data[i];
        }
    } else if (b_is_scalar) {
        // b is scalar, a is tensor
        float b_val = *b_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_data[i] - b_val;
        }
    } else {
        // Both are tensors - need to handle broadcasting properly
        // We'll iterate through the result tensor and compute indices for a and b
        std::vector<std::size_t> a_indices(a_shape.size(), 0);
        std::vector<std::size_t> b_indices(b_shape.size(), 0);
        std::vector<std::size_t> result_indices(result_shape.size(), 0);

        std::size_t linear_index = 0;
        while (linear_index < result.numel()) {
            // Compute a index (with broadcasting)
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                if (a_shape[i] == 1) {
                    a_indices[i] = 0;
                } else {
                    a_indices[i] = result_indices[i];
                }
            }

            // Compute b index (with broadcasting)
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                if (b_shape[i] == 1) {
                    b_indices[i] = 0;
                } else {
                    b_indices[i] = result_indices[i];
                }
            }

            // Compute offsets
            std::size_t a_offset = 0;
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                a_offset += a_indices[i];
                if (i < a_shape.size() - 1) {
                    a_offset *= a_shape[i + 1];
                }
            }

            std::size_t b_offset = 0;
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                b_offset += b_indices[i];
                if (i < b_shape.size() - 1) {
                    b_offset *= b_shape[i + 1];
                }
            }

            // Perform operation
            result_data[linear_index] = a_data[a_offset] - b_data[b_offset];

            // Increment result indices (like an odometer)
            for (std::size_t i = result_shape.size(); i > 0; --i) {
                std::size_t idx = i - 1;
                if (++result_indices[idx] < result_shape[idx]) {
                    break;
                }
                result_indices[idx] = 0;
            }

            ++linear_index;
        }
    }

    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();
    std::vector<std::size_t> b_shape = b.shape();

    // Handle broadcasting
    std::vector<std::size_t> result_shape;
    std::size_t max_dim = std::max(a_shape.size(), b_shape.size());

    // Pad shapes with 1s on the left (prepend dimensions)
    while (a_shape.size() < max_dim) {
        a_shape.insert(a_shape.begin(), 1);
    }
    while (b_shape.size() < max_dim) {
        b_shape.insert(b_shape.begin(), 1);
    }

    // Compute result shape and check for broadcasting compatibility
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_shape[i] == b_shape[i]) {
            result_shape.push_back(a_shape[i]);
        } else if (a_shape[i] == 1) {
            result_shape.push_back(b_shape[i]);
        } else if (b_shape[i] == 1) {
            result_shape.push_back(a_shape[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();
    const float* b_data = b.data();

    // Handle special cases for efficiency
    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (a_is_scalar && b_is_scalar) {
        // Both scalars
        *result_data = *a_data * *b_data;
    } else if (a_is_scalar) {
        // a is scalar, b is tensor
        float a_val = *a_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_val * b_data[i];
        }
    } else if (b_is_scalar) {
        // b is scalar, a is tensor
        float b_val = *b_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_data[i] * b_val;
        }
    } else {
        // Both are tensors - need to handle broadcasting properly
        // We'll iterate through the result tensor and compute indices for a and b
        std::vector<std::size_t> a_indices(a_shape.size(), 0);
        std::vector<std::size_t> b_indices(b_shape.size(), 0);
        std::vector<std::size_t> result_indices(result_shape.size(), 0);

        std::size_t linear_index = 0;
        while (linear_index < result.numel()) {
            // Compute a index (with broadcasting)
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                if (a_shape[i] == 1) {
                    a_indices[i] = 0;
                } else {
                    a_indices[i] = result_indices[i];
                }
            }

            // Compute b index (with broadcasting)
            for (std::size_t i = 0; i < b_shape.size(); ++i ) {
                if (b_shape[i] == 1) {
                    b_indices[i] = 0;
                } else {
                    b_indices[i] = result_indices[i];
                }
            }

            // Compute offsets
            std::size_t a_offset = 0;
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                a_offset += a_indices[i];
                if (i < a_shape.size() - 1) {
                    a_offset *= a_shape[i + 1];
                }
            }

            std::size_t b_offset = 0;
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                b_offset += b_indices[i];
                if (i < b_shape.size() - 1) {
                    b_offset *= b_shape[i + 1];
                }
            }

            // Perform operation
            result_data[linear_index] = a_data[a_offset] * b_data[b_offset];

            // Increment result indices (like an odometer)
            for (std::size_t i = result_shape.size(); i > 0; --i) {
                std::size_t idx = i - 1;
                if (++result_indices[idx] < result_shape[idx]) {
                    break;
                }
                result_indices[idx] = 0;
            }

            ++linear_index;
        }
    }

    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();
    std::vector<std::size_t> b_shape = b.shape();

    // Handle broadcasting
    std::vector<std::size_t> result_shape;
    std::size_t max_dim = std::max(a_shape.size(), b_shape.size());

    // Pad shapes with 1s on the left (prepend dimensions)
    while (a_shape.size() < max_dim) {
        a_shape.insert(a_shape.begin(), 1);
    }
    while (b_shape.size() < max_dim) {
        b_shape.insert(b_shape.begin(), 1);
    }

    // Compute result shape and check for broadcasting compatibility
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_shape[i] == b_shape[i]) {
            result_shape.push_back(a_shape[i]);
        } else if (a_shape[i] == 1) {
            result_shape.push_back(b_shape[i]);
        } else if (b_shape[i] == 1) {
            result_shape.push_back(a_shape[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();
    const float* b_data = b.data();

    // Handle special cases for efficiency
    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (a_is_scalar && b_is_scalar) {
        // Both scalars
        *result_data = *a_data / *b_data;
    } else if (a_is_scalar) {
        // a is scalar, b is tensor
        float a_val = *a_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_val / b_data[i];
        }
    } else if (b_is_scalar) {
        // b is scalar, a is tensor
        float b_val = *b_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = a_data[i] / b_val;
        }
    } else {
        // Both are tensors - need to handle broadcasting properly
        // We'll iterate through the result tensor and compute indices for a and b
        std::vector<std::size_t> a_indices(a_shape.size(), 0);
        std::vector<std::size_t> b_indices(b_shape.size(), 0);
        std::vector<std::size_t> result_indices(result_shape.size(), 0);

        std::size_t linear_index = 0;
        while (linear_index < result.numel()) {
            // Compute a index (with broadcasting)
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                if (a_shape[i] == 1) {
                    a_indices[i] = 0;
                } else {
                    a_indices[i] = result_indices[i];
                }
            }

            // Compute b index (with broadcasting)
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                if (b_shape[i] == 1) {
                    b_indices[i] = 0;
                } else {
                    b_indices[i] = result_indices[i];
                }
            }

            // Compute offsets
            std::size_t a_offset = 0;
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                a_offset += a_indices[i];
                if (i < a_shape.size() - 1) {
                    a_offset *= a_shape[i + 1];
                }
            }

            std::size_t b_offset = 0;
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                b_offset += b_indices[i];
                if (i < b_shape.size() - 1) {
                    b_offset *= b_shape[i + 1];
                }
            }

            // Perform operation
            result_data[linear_index] = a_data[a_offset] / b_data[b_offset];

            // Increment result indices (like an odometer)
            for (std::size_t i = result_shape.size(); i > 0; --i) {
                std::size_t idx = i - 1;
                if (++result_indices[idx] < result_shape[idx]) {
                    break;
                }
                result_indices[idx] = 0;
            }

            ++linear_index;
        }
    }

    return result;
}

Tensor pow(const Tensor& a, const Tensor& b) {
    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();
    std::vector<std::size_t> b_shape = b.shape();

    // Handle broadcasting
    std::vector<std::size_t> result_shape;
    std::size_t max_dim = std::max(a_shape.size(), b_shape.size());

    // Pad shapes with 1s on the left (prepend dimensions)
    while (a_shape.size() < max_dim) {
        a_shape.insert(a_shape.begin(), 1);
    }
    while (b_shape.size() < max_dim) {
        b_shape.insert(b_shape.begin(), 1);
    }

    // Compute result shape and check for broadcasting compatibility
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_shape[i] == b_shape[i]) {
            result_shape.push_back(a_shape[i]);
        } else if (a_shape[i] == 1) {
            result_shape.push_back(b_shape[i]);
        } else if (b_shape[i] == 1) {
            result_shape.push_back(a_shape[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();
    const float* b_data = b.data();

    // Handle special cases for efficiency
    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (a_is_scalar && b_is_scalar) {
        // Both scalars
        *result_data = std::pow(*a_data, *b_data);
    } else if (a_is_scalar) {
        // a is scalar, b is tensor
        float a_val = *a_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::pow(a_val, b_data[i]);
        }
    } else if (b_is_scalar) {
        // b is scalar, a is tensor
        float b_val = *b_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::pow(a_data[i], b_val);
        }
    } else {
        // Both are tensors - need to handle broadcasting properly
        // We'll iterate through the result tensor and compute indices for a and b
        std::vector<std::size_t> a_indices(a_shape.size(), 0);
        std::vector<std::size_t> b_indices(b_shape.size(), 0);
        std::vector<std::size_t> result_indices(result_shape.size(), 0);

        std::size_t linear_index = 0;
        while (linear_index < result.numel()) {
            // Compute a index (with broadcasting)
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                if (a_shape[i] == 1) {
                    a_indices[i] = 0;
                } else {
                    a_indices[i] = result_indices[i];
                }
            }

            // Compute b index (with broadcasting)
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                if (b_shape[i] == 1) {
                    b_indices[i] = 0;
                } else {
                    b_indices[i] = result_indices[i];
                }
            }

            // Compute offsets
            std::size_t a_offset = 0;
            for (std::size_t i = 0; i < a_shape.size(); ++i) {
                a_offset += a_indices[i];
                if (i < a_shape.size() - 1) {
                    a_offset *= a_shape[i + 1];
                }
            }

            std::size_t b_offset = 0;
            for (std::size_t i = 0; i < b_shape.size(); ++i) {
                b_offset += b_indices[i];
                if (i < b_shape.size() - 1) {
                    b_offset *= b_shape[i + 1];
                }
            }

            // Perform operation
            result_data[linear_index] = std::pow(a_data[a_offset], b_data[b_offset]);

            // Increment result indices (like an odometer)
            for (std::size_t i = result_shape.size(); i > 0; --i) {
                std::size_t idx = i - 1;
                if (++result_indices[idx] < result_shape[idx]) {
                    break;
                }
                result_indices[idx] = 0;
            }

            ++linear_index;
        }
    }

    return result;
}

Tensor softmax(const Tensor& a) {
    if (a.numel() == 0) {
        return a;
    }

    // For softmax, we always reduce along the last dimension
    std::size_t dim = a.dim() - 1;

    // Handle special case of 1D tensor
    if (a.dim() == 1) {
        // Numerically stable softmax: subtract max for stability
        float max_val = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < a.numel(); ++i) {
            max_val = std::max(max_val, a.data()[i]);
        }

        // Compute exponentials of (x - max)
        std::vector<float> exp_values;
        exp_values.reserve(a.numel());
        float sum_exp = 0.0f;
        for (std::size_t i = 0; i < a.numel(); ++i) {
            float exp_val = std::exp(a.data()[i] - max_val);
            exp_values.push_back(exp_val);
            sum_exp += exp_val;
        }

        // Normalize by sum of exponentials
        Tensor result(a.shape(), false);
        float* result_data = result.data();
        for (std::size_t i = 0; i < a.numel(); ++i) {
            result_data[i] = exp_values[i] / sum_exp;
        }

        return result;
    }

    // For multi-dimensional tensors, we reduce along the last dimension
    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();

    // Compute result shape (same as input for softmax)
    std::vector<std::size_t> result_shape = a_shape;

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Reduce along the last dimension
    std::size_t outer_size = 1;
    for (std::size_t i = 0; i < dim; ++i) {
        outer_size *= a_shape[i];
    }
    std::size_t inner_size = 1; // For last dimension, inner size is always 1
    std::size_t dim_size = a_shape[dim];

    for (std::size_t outer = 0; outer < outer_size; ++outer) {
        for (std::size_t inner = 0; inner < inner_size; ++inner) {
            // Find max value for numerical stability
            float max_val = std::numeric_limits<float>::lowest();
            for (std::size_t i = 0; i < dim_size; ++i) {
                // Compute index in input tensor
                std::size_t a_index = outer * (dim_size * inner_size) + i * inner_size + inner;
                max_val = std::max(max_val, a_data[a_index]);
            }

            // Compute exponentials of (x - max)
            std::vector<float> exp_values;
            exp_values.reserve(dim_size);
            float sum_exp = 0.0f;
            for (std::size_t i = 0; i < dim_size; ++i) {
                // Compute index in input tensor
                std::size_t a_index = outer * (dim_size * inner_size) + i * inner_size + inner;
                float exp_val = std::exp(a_data[a_index] - max_val);
                exp_values.push_back(exp_val);
                sum_exp += exp_val;
            }

            // Normalize by sum of exponentials
            for (std::size_t i = 0; i < dim_size; ++i) {
                // Compute index in input tensor
                std::size_t a_index = outer * (dim_size * inner_size) + i * inner_size + inner;
                // Compute index in result tensor (same as input for softmax)
                std::size_t result_index = a_index;
                result_data[result_index] = exp_values[i] / sum_exp;
            }
        }
    }

    return result;
}

Tensor log_softmax(const Tensor& a) {
    // Compute log_softmax(x) = log(softmax(x))
    Tensor softmax_result = softmax(a);
    // Apply log to each element
    Tensor result(softmax_result.shape(), false);
    float* result_data = result.data();
    const float* softmax_data = softmax_result.data();
    for (std::size_t i = 0; i < result.numel(); ++i) {
        result_data[i] = std::log(softmax_data[i]);
    }
    return result;
}

Tensor sum(const Tensor& a, std::size_t dim) {
    if (a.numel() == 0) {
        return a;
    }

    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();

    // Compute result shape (remove the reduction dimension)
    std::vector<std::size_t> result_shape;
    for (std::size_t i = 0; i < a_shape.size(); ++i) {
        if (i != dim) {
            result_shape.push_back(a_shape[i]);
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Reduce along the specified dimension
    std::size_t outer_size = 1;
    for (std::size_t i = 0; i < dim; ++i) {
        outer_size *= a_shape[i];
    }
    std::size_t inner_size = 1;
    for (std::size_t i = dim + 1; i < a_shape.size(); ++i) {
        inner_size *= a_shape[i];
    }
    std::size_t dim_size = a_shape[dim];

    for (std::size_t outer = 0; outer < outer_size; ++outer) {
        for (std::size_t inner = 0; inner < inner_size; ++inner) {
            float sum_val = 0.0f;
            for (std::size_t i = 0; i < dim_size; ++i) {
                // Compute index in input tensor
                std::size_t a_index = outer * (dim_size * inner_size) + i * inner_size + inner;
                sum_val += a_data[a_index];
            }
            // Compute index in result tensor
            std::size_t result_index = outer * inner_size + inner;
            result_data[result_index] = sum_val;
        }
    }

    return result;
}

Tensor mean(const Tensor& a, std::size_t dim) {
    if (a.numel() == 0) {
        return a;
    }

    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();

    // Compute result shape (remove the reduction dimension)
    std::vector<std::size_t> result_shape;
    for (std::size_t i = 0; i < a_shape.size(); ++i) {
        if (i != dim) {
            result_shape.push_back(a_shape[i]);
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Reduce along the specified dimension
    std::size_t outer_size = 1;
    for (std::size_t i = 0; i < dim; ++i) {
        outer_size *= a_shape[i];
    }
    std::size_t inner_size = 1;
    for (std::size_t i = dim + 1; i < a_shape.size(); ++i) {
        inner_size *= a_shape[i];
    }
    std::size_t dim_size = a_shape[dim];

    for (std::size_t outer = 0; outer < outer_size; ++outer) {
        for (std::size_t inner = 0; inner < inner_size; ++inner) {
            float sum_val = 0.0f;
            for (std::size_t i = 0; i < dim_size; ++i) {
                // Compute index in input tensor
                std::size_t a_index = outer * (dim_size * inner_size) + i * inner_size + inner;
                sum_val += a_data[a_index];
            }
            // Compute index in result tensor
            std::size_t result_index = outer * inner_size + inner;
            result_data[result_index] = sum_val / static_cast<float>(dim_size);
        }
    }

    return result;
}

Tensor max(const Tensor& a, std::size_t dim) {
    if (a.numel() == 0) {
        return a;
    }

    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();

    // Compute result shape (remove the reduction dimension)
    std::vector<std::size_t> result_shape;
    for (std::size_t i = 0; i < a_shape.size(); ++i) {
        if (i != dim) {
            result_shape.push_back(a_shape[i]);
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Reduce along the specified dimension
    std::size_t outer_size = 1;
    for (std::size_t i = 0; i < dim; ++i) {
        outer_size *= a_shape[i];
    }
    std::size_t inner_size = 1;
    for (std::size_t i = dim + 1; i < a_shape.size(); ++i) {
        inner_size *= a_shape[i];
    }
    std::size_t dim_size = a_shape[dim];

    for (std::size_t outer = 0; outer < outer_size; ++outer) {
        for (std::size_t inner = 0; inner < inner_size; ++inner) {
            float max_val = std::numeric_limits<float>::lowest();
            for (std::size_t i = 0; i < dim_size; ++i) {
                // Compute index in input tensor
                std::size_t a_index = outer * (dim_size * inner_size) + i * inner_size + inner;
                max_val = std::max(max_val, a_data[a_index]);
            }
            // Compute index in result tensor
            std::size_t result_index = outer * inner_size + inner;
            result_data[result_index] = max_val;
        }
    }

    return result;
}

Tensor min(const Tensor& a, std::size_t dim) {
    if (a.numel() == 0) {
        return a;
    }

    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Broadcast shapes to find result shape
    std::vector<std::size_t> a_shape = a.shape();

    // Compute result shape (remove the reduction dimension)
    std::vector<std::size_t> result_shape;
    for (std::size_t i = 0; i < a_shape.size(); ++i) {
        if (i != dim) {
            result_shape.push_back(a_shape[i]);
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Reduce along the specified dimension
    std::size_t outer_size = 1;
    for (std::size_t i = 0; i < dim; ++i) {
        outer_size *= a_shape[i];
    }
    std::size_t inner_size = 1;
    for (std::size_t i = dim + 1; i < a_shape.size(); ++i) {
        inner_size *= a_shape[i];
    }
    std::size_t dim_size = a_shape[dim];

    for (std::size_t outer = 0; outer < outer_size; ++outer) {
        for (std::size_t inner = 0; inner < inner_size; ++inner) {
            float min_val = std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < dim_size; ++i) {
                // Compute index in input tensor
                std::size_t a_index = outer * (dim_size * inner_size) + i * inner_size + inner;
                min_val = std::min(min_val, a_data[a_index]);
            }
            // Compute index in result tensor
            std::size_t result_index = outer * inner_size + inner;
            result_data[result_index] = min_val;
        }
    }

    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Check that both tensors are at least 2D
    if (a.dim() < 2 || b.dim() < 2) {
        throw std::invalid_argument("Matmul requires both tensors to be at least 2-dimensional");
    }

    // For matmul, we only support the last two dimensions for now
    // In a full implementation, we would handle batching, but for Phase 3 we keep it simple
    if (a.dim() > 2 || b.dim() > 2) {
        throw std::invalid_argument("Matmul with batching not yet implemented");
    }

    // Check inner dimensions for compatibility
    if (a.shape()[a.shape().size() - 1] != b.shape()[b.shape().size() - 2]) {
        throw std::invalid_argument("Incompatible dimensions for matmul");
    }

    // Result shape: [a.shape[0..-2], b.shape[-1]]
    std::vector<std::size_t> result_shape;
    for (std::size_t i = 0; i < a.shape().size() - 1; ++i) {
        result_shape.push_back(a.shape()[i]);
    }
    result_shape.push_back(b.shape()[b.shape().size() - 1]);

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();
    const float* b_data = b.data();

    // Perform matrix multiplication
    std::size_t a_rows = a.shape()[0];
    std::size_t a_cols = a.shape()[1];
    std::size_t b_cols = b.shape()[1];

    for (std::size_t i = 0; i < a_rows; ++i) {
        for (std::size_t j = 0; j < b_cols; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < a_cols; ++k) {
                std::size_t a_index = i * a_cols + k;
                std::size_t b_index = k * b_cols + j;
                sum += a_data[a_index] * b_data[b_index];
            }
            std::size_t result_index = i * b_cols + j;
            result_data[result_index] = sum;
        }
    }

    return result;
}

} // namespace autograd