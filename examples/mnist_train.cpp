#include "autograd/mnist_model.hpp"
#include "autograd/mnist_loader.hpp"
#include "autograd/optimizer.hpp"
#include "autograd/operations.hpp"
#include "autograd/internal.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace autograd {

/**
 * @brief Create mini-batches from tensors.
 * @param x Input tensor [N, 784]
 * @param y Target tensor [N, 10] (one-hot)
 * @param batch_size Batch size
 * @param shuffle Whether to shuffle indices
 * @param seed Random seed for shuffling
 * @return Vector of (x_batch, y_batch) pairs
 */
std::vector<std::pair<Tensor, Tensor>> create_batches(const Tensor& x, const Tensor& y, size_t batch_size, bool shuffle = true, unsigned int seed = 42) {
    size_t N = x.shape()[0];
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937 gen(seed);
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    std::vector<std::pair<Tensor, Tensor>> batches;

    for (size_t i = 0; i < N; i += batch_size) {
        size_t end = std::min(i + batch_size, N);
        size_t batch_N = end - i;

        // Create batch tensors
        Tensor x_batch = Tensor::empty({batch_N, 784}, false);
        Tensor y_batch = Tensor::empty({batch_N, 10}, false);

        const float* x_data = x.data();
        const float* y_data = y.data();
        float* x_batch_data = x_batch.data();
        float* y_batch_data = y_batch.data();

        for (size_t j = 0; j < batch_N; ++j) {
            size_t idx = indices[i + j];
            std::copy(x_data + idx * 784, x_data + (idx + 1) * 784, x_batch_data + j * 784);
            std::copy(y_data + idx * 10, y_data + (idx + 1) * 10, y_batch_data + j * 10);
        }

        batches.emplace_back(std::move(x_batch), std::move(y_batch));
    }

    return batches;
}

} // namespace autograd

int main() {
    using namespace autograd;
    using namespace std::chrono;

    std::cout << "==========================================" << std::endl;
    std::cout << "AutoGrad++ MNIST Training" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Configuration
    const size_t BATCH_SIZE = 64;
    const size_t EPOCHS = 20;
    const float LEARNING_RATE = 0.001f;  // Increased from 0.0001

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Epochs: " << EPOCHS << std::endl;
    std::cout << "  Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "  Optimizer: Adam" << std::endl;

    // Ensure MNIST data is downloaded
    std::cout << "\nEnsuring MNIST data is available..." << std::endl;
    if (!ensure_mnist_data("data/mnist")) {
        std::cerr << "Failed to ensure MNIST data" << std::endl;
        return 1;
    }

    // Load MNIST data
    std::cout << "\nLoading MNIST dataset..." << std::endl;
    auto start_load = high_resolution_clock::now();

    MNISTDataset train_dataset, test_dataset;
    try {
        train_dataset = load_mnist("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte");
        test_dataset = load_mnist("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::cerr << "Error loading MNIST: " << e.what() << std::endl;
        return 1;
    }

    auto end_load = high_resolution_clock::now();
    auto load_time = duration_cast<seconds>(end_load - start_load).count();
    std::cout << "Loaded in " << load_time << " seconds" << std::endl;
    std::cout << "  Train images: " << train_dataset.images.shape()[0] << std::endl;
    std::cout << "  Test images: " << test_dataset.images.shape()[0] << std::endl;

    // Labels are already one-hot from loader
    Tensor train_labels_onehot = train_dataset.labels;
    Tensor test_labels_onehot = test_dataset.labels;

    // Images are already normalized to [0, 1] from loader

    // Create model
    std::cout << "\nCreating MLP model (784 -> 256 -> 128 -> 10)..." << std::endl;
    MLP model(784, {256, 128}, 10, 42);
    std::cout << "  Total parameters: " << model.num_parameters() << std::endl;

    // Create optimizer
    auto optimizer = create_optimizer("adam", LEARNING_RATE);
    std::cout << "  Optimizer: " << optimizer->name() << std::endl;

    // Training loop
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Starting training..." << std::endl;
    std::cout << "==========================================" << std::endl;

    auto train_start = high_resolution_clock::now();

    for (size_t epoch = 1; epoch <= EPOCHS; ++epoch) {
        model.train();
        auto epoch_start = high_resolution_clock::now();

        // Create batches
        auto batches = create_batches(train_dataset.images, train_labels_onehot, BATCH_SIZE, true, 42 + epoch);

        float epoch_loss = 0.0f;
        size_t num_batches = 0;

        for (auto& batch : batches) {
            float loss = model.train_step(batch.first, batch.second, *optimizer);
            epoch_loss += loss;
            num_batches++;

            // Stop after first batch if loss is NaN
            if (std::isnan(loss)) {
                std::cout << "  WARNING: Loss is NaN after first batch, stopping early" << std::endl;
                break;
            }
        }

        epoch_loss /= num_batches;

        auto epoch_end = high_resolution_clock::now();
        auto epoch_time = duration_cast<milliseconds>(epoch_end - epoch_start).count();

        // Evaluate on test set
        model.eval();
        float test_acc = model.evaluate(test_dataset.images, test_labels_onehot);

        std::cout << "Epoch " << std::setw(2) << epoch << "/" << EPOCHS
                  << " | Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                  << " | Test Acc: " << std::setprecision(2) << test_acc << "%"
                  << " | Time: " << epoch_time << "ms" << std::endl;

        // Early stopping check - if we hit >95% we can stop early
        if (test_acc >= 95.0f) {
            std::cout << "\n*** Target accuracy (95%) reached! ***" << std::endl;
        }

        // Stop if loss is NaN
        if (std::isnan(epoch_loss)) {
            std::cout << "Stopping due to NaN loss" << std::endl;
            break;
        }
    }

    auto train_end = high_resolution_clock::now();
    auto total_time = duration_cast<seconds>(train_end - train_start).count();

    // Final evaluation
    model.eval();
    float final_acc = model.evaluate(test_dataset.images, test_labels_onehot);

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Total training time: " << total_time << " seconds" << std::endl;
    std::cout << "Final test accuracy: " << std::fixed << std::setprecision(2) << final_acc << "%" << std::endl;

    // Assert >95% accuracy
    if (final_acc >= 95.0f) {
        std::cout << "\n✓ SUCCESS: Achieved >95% test accuracy!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ FAILED: Did not achieve >95% test accuracy (got "
                  << std::fixed << std::setprecision(2) << final_acc << "%)" << std::endl;
        return 1;
    }
}