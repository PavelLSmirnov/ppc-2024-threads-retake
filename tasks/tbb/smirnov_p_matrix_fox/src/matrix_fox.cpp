#include "tbb/smirnov_p_matrix_fox/include/matrix_fox.hpp"

#include <atomic>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "tbb/tbb.h"

using std::cout;
using std::endl;

bool MatrixFoxAlgTaskTBB::validation() {
  internal_order_test();
  size_t input_count = taskData->inputs_count[0];
  n = static_cast<size_t>(round(sqrt(input_count)));
  bool valid = (taskData->inputs[0] != nullptr) && (taskData->inputs[1] != nullptr) &&
               (taskData->outputs[0] != nullptr) && (input_count == taskData->inputs_count[1]) &&
               (input_count == taskData->outputs_count[0]) && (n * n == input_count);
  if (!valid) {
    std::cerr << "Validation failed: " << std::endl;
    std::cerr << "input_count: " << input_count << std::endl;
    std::cerr << "inputs[0] != nullptr: " << (taskData->inputs[0] != nullptr) << std::endl;
    std::cerr << "inputs[1] != nullptr: " << (taskData->inputs[1] != nullptr) << std::endl;
    std::cerr << "outputs[0] != nullptr: " << (taskData->outputs[0] != nullptr) << std::endl;
    std::cerr << "inputs_count[0] == inputs_count[1]: " << (input_count == taskData->inputs_count[1]) << std::endl;
    std::cerr << "inputs_count[0] == outputs_count[0]: " << (input_count == taskData->outputs_count[0]) << std::endl;
    std::cerr << "n * n == input_count: " << (n * n == input_count) << std::endl;
  }
  return valid;
}

bool MatrixFoxAlgTaskTBB::pre_processing() {
  internal_order_test();
  int num_threads = 2;
  size_t matrix_size = taskData->inputs_count[0];
  if (matrix_size % num_threads != 0) {
    std::cerr << "Matrix size must be divisible by the number of threads\n";
    return false;
  }
  A = reinterpret_cast<double*>(taskData->inputs[0]);
  B = reinterpret_cast<double*>(taskData->inputs[1]);
  C = reinterpret_cast<double*>(taskData->outputs[0]);
  return true;
}

bool MatrixFoxAlgTaskTBB::run() {
  internal_order_test();
  try {
    const int num_threads = 2;
    const size_t block_size = n / num_threads;

    tbb::parallel_for(0, num_threads, [&](int thread_id) {
      const size_t start = thread_id * block_size;
      const size_t end = (thread_id == num_threads - 1) ? n : (thread_id + 1) * block_size;

      for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < n; ++j) {
          double sum = 0.0;
          for (size_t k = 0; k < n; ++k) {
            sum += A[i * n + k] * B[k * n + j];
          }
          C[i * n + j] = sum;
        }
      }
    });
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool MatrixFoxAlgTaskTBB::post_processing() {
  internal_order_test();
  return true;
}

void GetRandomValue(double* m, int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < size; i++) {
    m[i] = gen() % 100;
  }
}

void identityMatrix(double* m, int n, double k) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[i * n + j] = 0;
    }
    m[i * n + i] = k;
  }
}

void ModifidentityMatrix(double* m, int n, double k) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[i * n + (n - i - 1)] = 0;
    }
    m[i * n + (n - i - 1)] = k;
  }
}

std::vector<double> mulMatrix(const std::vector<double>& A, const std::vector<double>& B, int n) {
  if (n == 0) {
    return std::vector<double>();
  }
  std::vector<double> C(n * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
  return C;
}
