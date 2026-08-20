#include "autograd/operations.hpp"
#include <stdexcept>
#include <cmath>

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

Tensor pow(const Tensor& base, const Tensor& exponent) {
    (void)base;
    (void)exponent;
    throw std::runtime_error("pow not yet implemented");
}

// Unary operations
Tensor neg(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = -*a_data;
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = -a_data[i];
        }
    }

    return result;
}

Tensor exp(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = std::exp(*a_data);
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::exp(a_data[i]);
        }
    }

    return result;
}

Tensor log(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = std::log(*a_data);
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::log(a_data[i]);
        }
    }

    return result;
}

Tensor sqrt(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = std::sqrt(*a_data);
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::sqrt(a_data[i]);
        }
    }

    return result;
}

Tensor abs(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = std::fabs(*a_data);
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::fabs(a_data[i]);
        }
    }

    return result;
}

// Activation functions
Tensor relu(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = (*a_data > 0.0f) ? *a_data : 0.0f;
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = (a_data[i] > 0.0f) ? a_data[i] : 0.0f;
        }
    }

    return result;
}

Tensor sigmoid(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = 1.0f / (1.0f + std::exp(-*a_data));
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
        }
    }

    return result;
}

Tensor tanh(const Tensor& a) {
    // Broadcast shapes to find result shape (same as input for unary op)
    std::vector<std::size_t> result_shape = a.shape();

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* a_data = a.data();

    // Handle special case for efficiency
    bool a_is_scalar = (a.numel() == 1);

    if (a_is_scalar) {
        // Scalar tensor
        *result_data = std::tanh(*a_data);
    } else {
        // Tensor - iterate through all elements
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::tanh(a_data[i]);
        }
    }

    return result;
}

Tensor softmax(const Tensor& a) {
    // For softmax, we need to specify a dimension. Since PRD doesn't specify,
    // we'll implement a simple version that works on the last dimension.
    // But note: the PRD says "softmax (numerically stable)" without specifying dimension.
    // We'll assume it's applied to the last dimension as is common.
    // However, to keep it simple for now, we'll throw an exception and implement later.
    (void)a;
    throw std::runtime_error("softmax not yet implemented");
}

// Reductions
Tensor sum(const Tensor& a, std::size_t dim) {
    (void)a;
    (void)dim;
    throw std::runtime_error("sum not yet implemented");
}

Tensor mean(const Tensor& a, std::size_t dim) {
    (void)a;
    (void)dim;
    throw std::runtime_error("mean not yet implemented");
}

Tensor max(const Tensor& a, std::size_t dim) {
    (void)a;
    (void)dim;
    throw std::runtime_error("max not yet implemented");
}

Tensor min(const Tensor& a, std::size_t dim) {
    (void)a;
    (void)dim;
    throw std::runtime_error("min not yet implemented");
}

// Normalization
Tensor log_softmax(const Tensor& a) {
    (void)a;
    throw std::runtime_error("log_softmax not yet implemented");
}

// Matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b) {
    (void)a;
    (void)b;
    throw std::runtime_error("matmul not yet implemented");
}

} // namespace autograd