#include "autograd/mnist_loader.hpp"
#include "autograd/tensor.hpp"
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <cstring>

namespace autograd {

namespace {

// Read 32-bit integer in big-endian format (IDX format)
uint32_t read_be32(std::ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file) throw std::runtime_error("Failed to read 4 bytes");
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

// Download file using curl
bool download_file(const std::string& url, const std::string& filepath) {
    std::string cmd = "curl -L --fail --silent --show-error -o " + filepath + " " + url;
    int result = std::system(cmd.c_str());
    return result == 0;
}

} // anonymous namespace

MNISTDataset load_mnist(const std::string& images_path, const std::string& labels_path) {
    // Load images
    std::ifstream img_file(images_path, std::ios::binary);
    if (!img_file) {
        throw std::runtime_error("Cannot open images file: " + images_path);
    }

    uint32_t img_magic = read_be32(img_file);
    if (img_magic != 0x00000803) {
        throw std::runtime_error("Invalid images file magic number: " + std::to_string(img_magic));
    }

    uint32_t num_images = read_be32(img_file);
    uint32_t num_rows = read_be32(img_file);
    uint32_t num_cols = read_be32(img_file);

    if (num_rows != 28 || num_cols != 28) {
        throw std::runtime_error("Expected 28x28 images, got " + std::to_string(num_rows) + "x" + std::to_string(num_cols));
    }

    // Load labels
    std::ifstream lbl_file(labels_path, std::ios::binary);
    if (!lbl_file) {
        throw std::runtime_error("Cannot open labels file: " + labels_path);
    }

    uint32_t lbl_magic = read_be32(lbl_file);
    if (lbl_magic != 0x00000801) {
        throw std::runtime_error("Invalid labels file magic number: " + std::to_string(lbl_magic));
    }

    uint32_t num_labels = read_be32(lbl_file);
    if (num_images != num_labels) {
        throw std::runtime_error("Number of images (" + std::to_string(num_images) +
                                ") != number of labels (" + std::to_string(num_labels) + ")");
    }

    // Create tensors
    // Images: [N, 784] - flatten 28x28 and normalize to [0,1]
    Tensor images({static_cast<size_t>(num_images), 784}, false);
    // Labels: [N, 10] - one-hot encoded
    Tensor labels({static_cast<size_t>(num_images), 10}, false);

    // Read all image data
    std::vector<uint8_t> img_buffer(num_rows * num_cols);
    for (size_t i = 0; i < num_images; ++i) {
        img_file.read(reinterpret_cast<char*>(img_buffer.data()), img_buffer.size());
        if (!img_file) {
            throw std::runtime_error("Failed to read image " + std::to_string(i));
        }

        // Normalize to [0,1] and flatten
        float* img_ptr = images.data() + i * 784;
        for (size_t p = 0; p < 784; ++p) {
            img_ptr[p] = static_cast<float>(img_buffer[p]) / 255.0f;
        }
    }

    // Read all labels and one-hot encode
    for (size_t i = 0; i < num_labels; ++i) {
        uint8_t label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        if (!lbl_file) {
            throw std::runtime_error("Failed to read label " + std::to_string(i));
        }

        float* lbl_ptr = labels.data() + i * 10;
        std::fill(lbl_ptr, lbl_ptr + 10, 0.0f);
        if (label < 10) {
            lbl_ptr[label] = 1.0f;
        }
    }

    return {std::move(images), std::move(labels)};
}

std::vector<std::pair<Tensor, Tensor>> create_batches(
    const MNISTDataset& dataset,
    size_t batch_size,
    bool shuffle,
    unsigned int seed
) {
    size_t N = dataset.num_samples();
    if (N == 0) return {};

    // Create indices
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<std::pair<Tensor, Tensor>> batches;
    size_t num_batches = (N + batch_size - 1) / batch_size;

    for (size_t b = 0; b < num_batches; ++b) {
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, N);
        size_t actual_batch_size = end - start;

        Tensor batch_images({actual_batch_size, 784}, false);
        Tensor batch_labels({actual_batch_size, 10}, false);

        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t src_idx = indices[start + i];
            std::copy(
                dataset.images.data() + src_idx * 784,
                dataset.images.data() + (src_idx + 1) * 784,
                batch_images.data() + i * 784
            );
            std::copy(
                dataset.labels.data() + src_idx * 10,
                dataset.labels.data() + (src_idx + 1) * 10,
                batch_labels.data() + i * 10
            );
        }

        batches.emplace_back(std::move(batch_images), std::move(batch_labels));
    }

    return batches;
}

bool ensure_mnist_data(const std::string& data_dir) {
    namespace fs = std::filesystem;

    fs::create_directories(data_dir);

    const std::vector<std::pair<std::string, std::string>> files = {
        {"train-images-idx3-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"},
        {"train-labels-idx1-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"},
        {"t10k-images-idx3-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"},
        {"t10k-labels-idx1-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"}
    };

    bool all_exist = true;
    for (const auto& [filename, url] : files) {
        std::string filepath = data_dir + "/" + filename;
        if (!fs::exists(filepath)) {
            all_exist = false;
            std::string gz_path = filepath + ".gz";
            std::cout << "Downloading " << filename << "..." << std::endl;
            if (!download_file(url, gz_path)) {
                std::cerr << "Failed to download " << url << std::endl;
                return false;
            }
            // Decompress
            std::string cmd = "gzip -d " + gz_path;
            if (std::system(cmd.c_str()) != 0) {
                std::cerr << "Failed to decompress " << gz_path << std::endl;
                return false;
            }
        }
    }

    if (all_exist) {
        std::cout << "MNIST data already present in " << data_dir << std::endl;
    }

    return true;
}

} // namespace autograd