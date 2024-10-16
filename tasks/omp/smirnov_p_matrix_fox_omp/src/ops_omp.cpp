//  Copyright 2024 Smirnov Pavel
#include "omp/smirnov_p_matrix_fox_omp/include/ops_omp.hpp"

using SmirnovOMP::FoxBlockedParallel;
using SmirnovOMP::FoxBlockedSequential;
using SmirnovOMP::Matrix;

bool FoxBlockedSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool FoxBlockedSequential::pre_processing() {
  internal_order_test();
  auto* matrixA = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* matrixB = reinterpret_cast<double*>(taskData->inputs[1]);
  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  block_size = 32;
  A.resize(rows, std::vector<double>(cols));
  B.resize(rows, std::vector<double>(cols));
  C.resize(rows, std::vector<double>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      A[i][j] = matrixA[i * cols + j];
      B[i][j] = matrixB[i * cols + j];
    }
  }
  return true;
}

bool FoxBlockedSequential::run() {
  internal_order_test();
  for (std::vector<double>::size_type i = 0; i < A.size(); i += block_size) {
    for (std::vector<double>::size_type j = 0; j < B[0].size(); j += block_size) {
      for (std::vector<double>::size_type k = 0; k < A[0].size(); ++k) {
        for (std::vector<double>::size_type ii = i;
             ii < std::min(i + static_cast<std::vector<double>::size_type>(block_size), A.size()); ++ii) {
          for (std::vector<double>::size_type jj = j;
               jj < std::min(j + static_cast<std::vector<double>::size_type>(block_size), B[0].size()); ++jj) {
            C[ii][jj] += A[ii][k] * B[k][jj];
          }
        }
      }
    }
  }
  return true;
}

bool FoxBlockedSequential::post_processing() {
  internal_order_test();
  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (std::vector<double>::size_type i = 0; i < C.size(); ++i) {
    for (std::vector<double>::size_type j = 0; j < C[0].size(); ++j) {
      out_ptr[i * C[0].size() + j] = C[i][j];
    }
  }
  return true;
}

bool FoxBlockedParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool FoxBlockedParallel::pre_processing() {
  internal_order_test();
  auto* matrixA = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* matrixB = reinterpret_cast<double*>(taskData->inputs[1]);
  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  block_size = 32;
  A.resize(rows, std::vector<double>(cols));
  B.resize(rows, std::vector<double>(cols));
  C.resize(rows, std::vector<double>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      A[i][j] = matrixA[i * cols + j];
      B[i][j] = matrixB[i * cols + j];
    }
  }
  return true;
}

bool FoxBlockedParallel::run() {
  internal_order_test();

#pragma omp parallel for
  for (int ii = 0; ii < static_cast<int>(A.size()); ii += block_size) {
    for (std::vector<double>::size_type jj = 0; jj < B[0].size();
         jj += static_cast<std::vector<double>::size_type>(block_size)) {
      for (std::vector<double>::size_type k = 0; k < A[0].size(); ++k) {
        for (int i = ii; i < std::min(ii + block_size, static_cast<int>(A.size())); ++i) {
          for (std::vector<double>::size_type j = jj;
               j < std::min(jj + static_cast<std::vector<double>::size_type>(block_size), B[0].size()); ++j) {
            C[i][j] += A[i][k] * B[k][j];
          }
        }
      }
    }
  }

  return true;
}

bool FoxBlockedParallel::post_processing() {
  internal_order_test();
  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (std::vector<double>::size_type i = 0; i < C.size(); ++i) {
    for (std::vector<double>::size_type j = 0; j < C[0].size(); ++j) {
      out_ptr[i * C[0].size() + j] = C[i][j];
    }
  }
  return true;
}
