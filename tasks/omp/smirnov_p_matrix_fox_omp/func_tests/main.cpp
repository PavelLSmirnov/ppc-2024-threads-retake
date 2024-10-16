//  Copyright 2024 Smirnov Pavel
#include <gtest/gtest.h>

#include <vector>

#include "omp/smirnov_p_matrix_fox_omp/include/ops_omp.hpp"

using namespace SmirnovOMP;

TEST(FoxBlockedParallel, MatrixMultiplication2x2) {
  // Define input matrices
  std::vector<double> matrixA = {1, 2, 3, 4};
  std::vector<double> matrixB = {4, 3, 2, 1};
  std::vector<double> expectedOutput = {8, 5, 20, 13};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  taskData->inputs_count.emplace_back(2);
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB.data()));
  taskData->inputs_count.emplace_back(2);
  taskData->inputs_count.emplace_back(2);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(new double[4]()));
  taskData->outputs_count.emplace_back(2);
  taskData->outputs_count.emplace_back(2);
  // Create Task
  FoxBlockedParallel foxBlockedParallel(taskData);
  foxBlockedParallel.validation();
  foxBlockedParallel.pre_processing();
  auto *output = reinterpret_cast<double *>(taskData->outputs[0]);
  foxBlockedParallel.run();
  foxBlockedParallel.post_processing();
  // Check the output
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_DOUBLE_EQ(output[i], expectedOutput[i]);
  }
  // Free memory
  delete[] output;
}

TEST(FoxBlockedParallel, MatrixMultiplication) {
  // Define input matrices
  std::vector<double> matrixA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> matrixB = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> expectedOutput = {30, 24, 18, 84, 69, 54, 138, 114, 90};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  taskData->inputs_count.emplace_back(3);
  taskData->inputs_count.emplace_back(3);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB.data()));
  taskData->inputs_count.emplace_back(3);
  taskData->inputs_count.emplace_back(3);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(new double[9]()));
  taskData->outputs_count.emplace_back(3);
  taskData->outputs_count.emplace_back(3);
  // Create Task
  FoxBlockedParallel foxBlockedParallel(taskData);
  foxBlockedParallel.validation();
  foxBlockedParallel.pre_processing();
  auto *output = reinterpret_cast<double *>(taskData->outputs[0]);
  foxBlockedParallel.run();
  foxBlockedParallel.post_processing();
  // Check the output
  for (size_t i = 0; i < 9; ++i) {
    ASSERT_DOUBLE_EQ(output[i], expectedOutput[i]);
  }
  // Free memory
  delete[] output;
}

TEST(FoxBlockedParallel, MatrixMultiplication_VerySmallMatrices) {
  // Define input matrices
  std::vector<double> matrixA(10 * 10);
  std::vector<double> matrixB(10 * 10);
  std::vector<double> expectedOutput(10 * 10);
  // Initialize matrices with random values
  for (size_t i = 0; i < 10 * 10; ++i) {
    matrixA[i] = rand() % 10;
    matrixB[i] = rand() % 10;
    expectedOutput[i] = 0;
  }
  for (size_t i = 0; i < 10; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      for (size_t k = 0; k < 10; ++k) {
        expectedOutput[i * 10 + j] += matrixA[i * 10 + k] * matrixB[k * 10 + j];
      }
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  taskData->inputs_count.emplace_back(10);
  taskData->inputs_count.emplace_back(10);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB.data()));
  taskData->inputs_count.emplace_back(10);
  taskData->inputs_count.emplace_back(10);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(new double[10 * 10]()));
  taskData->outputs_count.emplace_back(10);
  taskData->outputs_count.emplace_back(10);
  // Create Task
  FoxBlockedParallel foxBlockedParallel(taskData);
  foxBlockedParallel.validation();
  foxBlockedParallel.pre_processing();
  auto *output = reinterpret_cast<double *>(taskData->outputs[0]);
  foxBlockedParallel.run();
  foxBlockedParallel.post_processing();
  // Check the output
  for (size_t i = 0; i < 10 * 10; ++i) {
    ASSERT_DOUBLE_EQ(output[i], expectedOutput[i]);
  }
  // Free memory
  delete[] output;
}

TEST(FoxBlockedParallel, MatrixMultiplication_SmallMatrices) {
  // Define input matrices
  std::vector<double> matrixA(100 * 100);
  std::vector<double> matrixB(100 * 100);
  std::vector<double> expectedOutput(100 * 100);
  // Initialize matrices with random values
  for (size_t i = 0; i < 100 * 100; ++i) {
    matrixA[i] = rand() % 10;
    matrixB[i] = rand() % 10;
    expectedOutput[i] = 0;
  }
  for (size_t i = 0; i < 100; ++i) {
    for (size_t j = 0; j < 100; ++j) {
      for (size_t k = 0; k < 100; ++k) {
        expectedOutput[i * 100 + j] += matrixA[i * 100 + k] * matrixB[k * 100 + j];
      }
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  taskData->inputs_count.emplace_back(100);
  taskData->inputs_count.emplace_back(100);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB.data()));
  taskData->inputs_count.emplace_back(100);
  taskData->inputs_count.emplace_back(100);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(new double[100 * 100]()));
  taskData->outputs_count.emplace_back(100);
  taskData->outputs_count.emplace_back(100);
  // Create Task
  FoxBlockedParallel foxBlockedParallel(taskData);
  foxBlockedParallel.validation();
  foxBlockedParallel.pre_processing();
  auto *output = reinterpret_cast<double *>(taskData->outputs[0]);
  foxBlockedParallel.run();
  foxBlockedParallel.post_processing();
  // Check the output
  for (size_t i = 0; i < 100 * 100; ++i) {
    ASSERT_DOUBLE_EQ(output[i], expectedOutput[i]);
  }
  // Free memory
  delete[] output;
}

TEST(FoxBlockedParallel, MatrixMultiplicationWithNegatives) {
  // Define input matrices with negative values
  std::vector<double> matrixA = {-1, -2, -3, -4, -5, -6, -7, -8, -9};
  std::vector<double> matrixB = {-9, -8, -7, -6, -5, -4, -3, -2, -1};
  std::vector<double> expectedOutput = {30, 24, 18, 84, 69, 54, 138, 114, 90};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  taskData->inputs_count.emplace_back(3);
  taskData->inputs_count.emplace_back(3);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB.data()));
  taskData->inputs_count.emplace_back(3);
  taskData->inputs_count.emplace_back(3);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(new double[9]()));
  taskData->outputs_count.emplace_back(3);
  taskData->outputs_count.emplace_back(3);
  // Create Task
  FoxBlockedParallel foxBlockedParallel(taskData);
  foxBlockedParallel.validation();
  foxBlockedParallel.pre_processing();
  auto *output = reinterpret_cast<double *>(taskData->outputs[0]);
  foxBlockedParallel.run();
  foxBlockedParallel.post_processing();
  // Check the output
  for (size_t i = 0; i < 9; ++i) {
    ASSERT_DOUBLE_EQ(output[i], expectedOutput[i]);
  }
  // Free memory
  delete[] output;
}
