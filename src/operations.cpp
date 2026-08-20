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
    // Broadcast shapes to find result shape
    std::vector<std::size_t> base_shape = base.shape();
    std::vector<std::size_t> exp_shape = exponent.shape();

    // Handle broadcasting
    std::vector<std::size_t> result_shape;
    std::size_t max_dim = std::max(base_shape.size(), exp_shape.size());

    // Pad shapes with 1s on the left (prepend dimensions)
    while (base_shape.size() < max_dim) {
        base_shape.insert(base_shape.begin(), 1);
    }
    while (exp_shape.size() < max_dim) {
        exp_shape.insert(exp_shape.begin(), 1);
    }

    // Compute result shape and check for broadcasting compatibility
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (base_shape[i] == exp_shape[i]) {
            result_shape.push_back(base_shape[i]);
        } else if (base_shape[i] == 1) {
            result_shape.push_back(exp_shape[i]);
        } else if (exp_shape[i] == 1) {
            result_shape.push_back(base_shape[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting");
        }
    }

    // Create result tensor
    Tensor result(result_shape, false);

    // Early return for empty tensels
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    float* result_data = result.data();
    const float* base_data = base.data();
    const float* exp_data = exponent.data();

    // Handle special cases for efficiency
    bool base_is_scalar = (base.numel() == 1);
    bool exp_is_scalar = (exponent.numel() == 1);

    if (base_is_scalar && exp_is_scalar) {
        // Both scalars
        *result_data = std::pow(*base_data, *exp_data);
    } else if (base_is_scalar) {
        // base is scalar, exponent is tensor
        float base_val = *base_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::pow(base_val, exp_data[i]);
        }
    } else if (exp_is_scalar) {
        // exponent is scalar, base is tensor
        float exp_val = *exp_data;
        for (std::size_t i = 0; i < result.numel(); ++i) {
            result_data[i] = std::pow(base_data[i], exp_val);
        }
    } else {
        // Both are tensors - need to handle broadcasting properly
        // We'll iterate through the result tensor and compute indices for base and exponent
        std::vector<std::size_t> base_indices(base_shape.size(), 0);
        std::vector<std::size_t> exp_indices(exp_shape.size(), 0);
        std::vector<std::size_t> result_indices(result_shape.size(), 0);

        std::size_t linear_index = 0;
        while (linear_index < result.numel()) {
            // Compute base index (with broadcasting)
            for (std::size_t i = 0; i < base_shape.size(); ++i) {
                if (base_shape[i] == 1) {
                    base_indices[i] = 0;
                } else {
                    base_indices[i] = result_indices[i];
                }
            }

            // Compute exponent index (with broadcasting)
            for (std::size_t i = 0; i < exp_shape.size(); ++i) {
                if (exp_shape[i] == 1) {
                    exp_indices[i] = 0;
                } else {
                    exp_indices[i] = result_indices[i];
                }
            }

            // Compute offsets
            std::size_t base_offset = 0;
            for (std::size_t i = 0; i < base_shape.size(); ++i) {
                base_offset += base_indices[i];
                if (i < base_shape.size() - 1) {
                    base_offset *= base_shape[i + 1];
                }
            }

            std::size_t exp_offset = 0;
            for (std::size_t i = 0; i < exp_shape.size(); ++i) {
                exp_offset += exp_indices[i];
                if (i < exp_shape.size() - 1) {
                    exp_offset *= exp_shape[i + 1];
                }
            }

            // Perform operation
            result_data[linear_index] = std::pow(base_data[base_offset], exp_data[exp_offset]);

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
    // For numerical stability, we compute softmax as:
    // softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    // where max(x) is the maximum value in the slice along the last dimension

    // We'll apply softmax to the last dimension as is common convention
    // std::size_t last_dim = a.dim() > 0 ? a.dim() - 1 : 0;

    // Get shape and compute outer size (product of all dimensions except last)
    std::vector<std::size_t> shape = a.shape();
    std::size_t outer_size = 1;
    for (std::size_t i = 0; i < shape.size() - 1; ++i) {
        outer_size *= shape[i];
    }
    std::size_t inner_size = shape.back(); // size of last dimension

    // Handle edge cases
    if (outer_size == 0 || inner_size == 0) {
        Tensor result(shape, false);
        return result;
    }

    // Create result tensor with same shape
    Tensor result(shape, false);

    // Get data pointers
    const float* input_data = a.data();
    float* output_data = result.data();

    // Process each slice along the last dimension
    for (std::size_t outer = 0; outer < outer_size; ++outer) {
        // Find max value in this slice
        float max_val = input_data[outer * inner_size]; // Initialize with first element
        for (std::size_t i = 1; i < inner_size; ++i) {
            float val = input_data[outer * inner_size + i];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute exponentials and their sum
        float exp_sum = 0.0f;
        for (std::size_t i = 0; i < inner_size; ++i) {
            float val = input_data[outer * inner_size + i];
            float exp_val = std::exp(val - max_val);
            output_data[outer * inner_size + i] = exp_val;
            exp_sum += exp_val;
        }

        // Normalize by dividing by the sum
        for (std::size_t i = 0; i < inner_size; ++i) {
            output_data[outer * inner_size + i] /= exp_sum;
        }
    }

    return result;
}

// Reductions
Tensor sum(const Tensor& a, std::size_t dim) {
    // Validate dimension
    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Compute output shape: same as input but with dim dimension removed
    // (following NumPy convention where sum over size-0 dimension removes it)
    std::vector<std::size_t> input_shape = a.shape();
    std::vector<std::size_t> output_shape;
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (i != dim) {
            output_shape.push_back(input_shape[i]);
        }
    }

    // Create result tensor
    Tensor result(output_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    const float* input_data = a.data();
    float* output_data = result.data();

    // Compute strides for efficient indexing
    std::vector<std::ptrdiff_t> input_strides_pt = a.impl()->get_strides();
    std::vector<std::size_t> input_strides(input_strides_pt.begin(), input_strides_pt.end());

    // Size of the dimension we're summing over
    std::size_t dim_size = input_shape[dim];

    // For each output element, compute the sum along the specified dimension
    std::size_t total_output_elements = result.numel();
    for (std::size_t i = 0; i < total_output_elements; ++i) {
        // Compute the offset in the input tensor for this output element
        std::size_t input_offset = 0;
        std::size_t remaining = i;

        // Compute coordinates in output tensor
        std::vector<std::size_t> output_coords(output_shape.size());
        for (int j = static_cast<int>(output_shape.size()) - 1; j >= 0; --j) {
            output_coords[j] = remaining % output_shape[j];
            remaining /= output_shape[j];
        }

        // Map output coordinates to input coordinates (skipping the reduced dimension)
        std::size_t input_offset_mapped = 0;
        std::size_t in_idx = 0;
        for (std::size_t j = 0; j < input_shape.size(); ++j) {
            if (j == dim) {
                // Skip the reduced dimension
                continue;
            }
            input_offset_mapped += output_coords[in_idx] * input_strides[j];
            in_idx++;
        }

        // Now compute the sum along the specified dimension for this output location
        float sum_val = 0.0f;
        for (std::size_t j = 0; j < dim_size; ++j) {
            // Compute input offset: base_offset + j * stride_of_dim_in_input
            std::size_t offset = input_offset_mapped + j * input_strides[dim];
            sum_val += input_data[offset];
        }

        output_data[i] = sum_val;
    }

    return result;
}

Tensor mean(const Tensor& a, std::size_t dim) {
    // Validate dimension
    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Handle zero-sized dimension case
    std::size_t dim_size = a.shape()[dim];
    if (dim_size == 0) {
        // For zero-sized dimension, mean is undefined 
        // Following NumPy, we return NaN for mean over zero-sized dimension
        std::vector<std::size_t> output_shape;
        for (std::size_t i = 0; i < a.shape().size(); ++i) {
            if (i != dim) {
                output_shape.push_back(a.shape()[i]);
            }
        }
        Tensor result(output_shape, false);
        float* result_data = result.data();
        *result_data = std::numeric_limits<float>::quiet_NaN();
        return result;
    }

    // Compute sum along the dimension
    Tensor sum_result = sum(a, dim);

    // Create result tensor with same shape as sum_result
    Tensor result(sum_result.shape(), false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    const float* sum_data = sum_result.data();
    float* result_data = result.data();

    // Divide each element by the dimension size
    for (std::size_t i = 0; i < result.numel(); ++i) {
        result_data[i] = sum_data[i] / static_cast<float>(dim_size);
    }

    return result;
}

Tensor max(const Tensor& a, std::size_t dim) {
    // Validate dimension
    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Handle zero-sized dimension case
    std::size_t dim_size = a.shape()[dim];
    if (dim_size == 0) {
        // For zero-sized dimension, max is undefined
        // Following NumPy convention, we return NaN or raise an error
        // For simplicity, we'll return a tensor with appropriate shape filled with -infinity
        // But actually, let's follow the pattern of returning a tensor with the dimension removed
        std::vector<std::size_t> output_shape;
        for (std::size_t i = 0; i < a.shape().size(); ++i) {
            if (i != dim) {
                output_shape.push_back(a.shape()[i]);
            }
        }
        Tensor result(output_shape, false);
        // Fill with -infinity to indicate undefined max
        if (result.numel() > 0) {
            std::fill(result.data(), result.data() + result.numel(),
                     -std::numeric_limits<float>::infinity());
        }
        return result;
    }

    // Compute output shape: same as input but with dim dimension removed
    std::vector<std::size_t> input_shape = a.shape();
    std::vector<std::size_t> output_shape;
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (i != dim) {
            output_shape.push_back(input_shape[i]);
        }
    }

    // Create result tensor
    Tensor result(output_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    const float* input_data = a.data();
    float* output_data = result.data();

    // Compute strides for efficient indexing
    std::vector<std::ptrdiff_t> input_strides_pt = a.impl()->get_strides();
    std::vector<std::size_t> input_strides(input_strides_pt.begin(), input_strides_pt.end());

    // For each output element, compute the max along the specified dimension
    std::size_t total_output_elements = result.numel();
    for (std::size_t i = 0; i < total_output_elements; ++i) {
        // Compute the offset in the input tensor for this output element
        std::size_t input_offset = 0;
        std::size_t remaining = i;

        // Compute coordinates in output tensor
        std::vector<std::size_t> output_coords(output_shape.size());
        for (int j = static_cast<int>(output_shape.size()) - 1; j >= 0; --j) {
            output_coords[j] = remaining % output_shape[j];
            remaining /= output_shape[j];
        }

        // Map output coordinates to input coordinates (skipping the reduced dimension)
        std::size_t input_offset_mapped = 0;
        std::size_t in_idx = 0;
        for (std::size_t j = 0; j < input_shape.size(); ++j) {
            if (j == dim) {
                // Skip the reduced dimension
                continue;
            }
            input_offset_mapped += output_coords[in_idx] * input_strides[j];
            in_idx++;
        }

        // Now compute the max along the specified dimension for this output location
        float max_val = -std::numeric_limits<float>::infinity(); // Start with negative infinity
        for (std::size_t j = 0; j < dim_size; ++j) {
            // Compute input offset: base_offset + j * stride_of_dim_in_input
            std::size_t offset = input_offset_mapped + j * input_strides[dim];
            float val = input_data[offset];
            if (val > max_val) {
                max_val = val;
            }
        }

        output_data[i] = max_val;
    }

    return result;
}

Tensor min(const Tensor& a, std::size_t dim) {
    // Validate dimension
    if (dim >= a.dim()) {
        throw std::invalid_argument("Dimension out of range");
    }

    // Handle zero-sized dimension case
    std::size_t dim_size = a.shape()[dim];
    if (dim_size == 0) {
        // For zero-sized dimension, min is undefined
        // Following NumPy convention, we return NaN or raise an error
        // For simplicity, we'll return a tensor with appropriate shape filled with +infinity
        // But actually, let's follow the pattern of returning a tensor with the dimension removed
        std::vector<std::size_t> output_shape;
        for (std::size_t i = 0; i < a.shape().size(); ++i) {
            if (i != dim) {
                output_shape.push_back(a.shape()[i]);
            }
        }
        Tensor result(output_shape, false);
        // Fill with +infinity to indicate undefined min
        if (result.numel() > 0) {
            std::fill(result.data(), result.data() + result.numel(),
                     std::numeric_limits<float>::infinity());
        }
        return result;
    }

    // Compute output shape: same as input but with dim dimension removed
    std::vector<std::size_t> input_shape = a.shape();
    std::vector<std::size_t> output_shape;
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (i != dim) {
            output_shape.push_back(input_shape[i]);
        }
    }

    // Create result tensor
    Tensor result(output_shape, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    const float* input_data = a.data();
    float* output_data = result.data();

    // Compute strides for efficient indexing
    std::vector<std::ptrdiff_t> input_strides_pt = a.impl()->get_strides();
    std::vector<std::size_t> input_strides(input_strides_pt.begin(), input_strides_pt.end());

    // For each output element, compute the min along the specified dimension
    std::size_t total_output_elements = result.numel();
    for (std::size_t i = 0; i < total_output_elements; ++i) {
        // Compute the offset in the input tensor for this output element
        std::size_t input_offset = 0;
        std::size_t remaining = i;

        // Compute coordinates in output tensor
        std::vector<std::size_t> output_coords(output_shape.size());
        for (int j = static_cast<int>(output_shape.size()) - 1; j >= 0; --j) {
            output_coords[j] = remaining % output_shape[j];
            remaining /= output_shape[j];
        }

        // Map output coordinates to input coordinates (skipping the reduced dimension)
        std::size_t input_offset_mapped = 0;
        std::size_t in_idx = 0;
        for (std::size_t j = 0; j < input_shape.size(); ++j) {
            if (j == dim) {
                // Skip the reduced dimension
                continue;
            }
            input_offset_mapped += output_coords[in_idx] * input_strides[j];
            in_idx++;
        }

        // Now compute the min along the specified dimension for this output location
        float min_val = std::numeric_limits<float>::infinity(); // Start with positive infinity
        for (std::size_t j = 0; j < dim_size; ++j) {
            // Compute input offset: base_offset + j * stride_of_dim_in_input
            std::size_t offset = input_offset_mapped + j * input_strides[dim];
            float val = input_data[offset];
            if (val < min_val) {
                min_val = val;
            }
        }

        output_data[i] = min_val;
    }

    return result;
}

// Normalization
Tensor log_softmax(const Tensor& a) {
    // For numerical stability, we compute log_softmax as:
    // log_softmax(x_i) = (x_i - max(x)) - log(sum_j exp(x_j - max(x)))
    // where max(x) is the maximum value in the slice along the last dimension

    // We'll apply log_softmax to the last dimension as is common convention
    // std::size_t last_dim = a.dim() > 0 ? a.dim() - 1 : 0;

    // Get shape and compute outer size (product of all dimensions except last)
    std::vector<std::size_t> shape = a.shape();
    std::size_t outer_size = 1;
    for (std::size_t i = 0; i < shape.size() - 1; ++i) {
        outer_size *= shape[i];
    }
    std::size_t inner_size = shape.back(); // size of last dimension

    // Handle edge cases
    if (outer_size == 0 || inner_size == 0) {
        Tensor result(shape, false);
        return result;
    }

    // Create result tensor with same shape
    Tensor result(shape, false);

    // Get data pointers
    const float* input_data = a.data();
    float* output_data = result.data();

    // Process each slice along the last dimension
    for (std::size_t outer = 0; outer < outer_size; ++outer) {
        // Find max value in this slice
        float max_val = input_data[outer * inner_size]; // Initialize with first element
        for (std::size_t i = 1; i < inner_size; ++i) {
            float val = input_data[outer * inner_size + i];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute exponentials and their sum
        float exp_sum = 0.0f;
        for (std::size_t i = 0; i < inner_size; ++i) {
            float val = input_data[outer * inner_size + i];
            float exp_val = std::exp(val - max_val);
            exp_sum += exp_val;
        }

        // Compute log of the sum
        float log_exp_sum = std::log(exp_sum);

        // Compute log_softmax: (x_i - max_val) - log_sum_exp
        for (std::size_t i = 0; i < inner_size; ++i) {
            float val = input_data[outer * inner_size + i];
            output_data[outer * inner_size + i] = (val - max_val) - log_exp_sum;
        }
    }

    return result;
}

// Matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b) {
    // Validate that both tensors are contiguous
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::invalid_argument("matmul only supports contiguous tensors");
    }

    // Validate that both tensors are rank-2 (matrices)
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::invalid_argument("matmul only supports rank-2 tensors (matrices)");
    }

    // Get dimensions
    std::size_t M = a.shape()[0]; // rows of a
    std::size_t K = a.shape()[1]; // columns of a
    std::size_t K_b = b.shape()[0]; // rows of b
    std::size_t N = b.shape()[1]; // columns of b

    // Validate inner dimensions match for matrix multiplication
    if (K != K_b) {
        throw std::invalid_argument("Inner dimensions must agree for matmul");
    }

    // Create result tensor with shape [M, N]
    Tensor result({M, N}, false);

    // Early return for empty tensors
    if (result.numel() == 0) {
        return result;
    }

    // Get data pointers
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();

    // Perform matrix multiplication: C = A * B
    // C[i, j] = sum_k (A[i, k] * B[k, j])
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }

    return result;
}

} // namespace autograd