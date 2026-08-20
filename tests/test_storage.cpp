#include "autograd/storage.hpp"
#include <gtest/gtest.h>

namespace autograd {

TEST(StorageTest, DefaultConstruction) {
    Storage storage(0);
    EXPECT_EQ(storage.size(), 0);
    EXPECT_EQ(storage.data(), nullptr);
}

TEST(StorageTest, AllocationAndDeallocation) {
    const std::size_t size = 100;
    Storage storage(size);
    EXPECT_EQ(storage.size(), size);
    EXPECT_NE(storage.data(), nullptr);

    // Check that we can write to the memory
    float* data = storage.data();
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Check that we can read back
    for (std::size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], static_cast<float>(i));
    }
}

TEST(StorageTest, ZeroSizeStorage) {
    Storage storage(0);
    EXPECT_EQ(storage.size(), 0);
    EXPECT_EQ(storage.data(), nullptr);
}

TEST(StorageTest, OverflowProtection) {
    // Test size that would cause overflow in size * sizeof(float)
    const std::size_t max_size = (std::size_t(-1) / sizeof(float)) + 1;
    EXPECT_THROW(Storage storage(max_size), std::invalid_argument);

    // Test size that would cause overflow in bytes + alignment padding
    const std::size_t max_bytes = std::size_t(-1) - 63;
    const std::size_t problematic_size = (max_bytes / sizeof(float)) + 1;
    EXPECT_THROW(Storage storage(problematic_size), std::invalid_argument);
}

} // namespace autograd