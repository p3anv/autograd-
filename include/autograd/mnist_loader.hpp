#pragma once

#include "autograd/tensor.hpp"
#include <string>
#include <vector>
#include <optional>

namespace autograd {

/**
 * @brief MNIST dataset container.
 */
struct MNISTDataset {
    Tensor images;  // [N, 784] normalized to [0, 1]
    Tensor labels;  // [N, 10] one-hot encoded
    size_t num_samples() const { return images.shape()[0]; }
};

/**
 * @brief Load MNIST dataset from IDX files.
 *
 * Expected files:
 * - train-images-idx3-ubyte (60000 x 28 x 28)
 * - train-labels-idx1-ubyte (60000)
 * - t10k-images-idx3-ubyte (10000 x 28 x 28)
 * - t10k-labels-idx1-ubyte (10000)
 *
 * @param images_path Path to images IDX file
 * @param labels_path Path to labels IDX file
 * @return MNISTDataset with images normalized to [0,1] and labels one-hot encoded
 */
MNISTDataset load_mnist(const std::string& images_path, const std::string& labels_path);

/**
 * @brief Create mini-batches from dataset.
 *
 * @param dataset Source dataset
 * @param batch_size Size of each batch
 * @param shuffle Whether to shuffle indices (default: true)
 * @param seed Random seed for shuffling (default: 42)
 * @return Vector of (images_batch, labels_batch) pairs
 */
std::vector<std::pair<Tensor, Tensor>> create_batches(
    const MNISTDataset& dataset,
    size_t batch_size,
    bool shuffle = true,
    unsigned int seed = 42
);

/**
 * @brief Download MNIST files if not present.
 *
 * @param data_dir Directory to store/find MNIST files
 * @return true if all files present or downloaded successfully
 */
bool ensure_mnist_data(const std::string& data_dir = "data/mnist");

} // namespace autograd