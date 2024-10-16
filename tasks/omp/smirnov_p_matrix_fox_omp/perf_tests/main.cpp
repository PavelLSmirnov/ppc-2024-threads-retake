//  Copyright 2024 Smirnov Pavel
#include <gtest/gtest.h>
#include <omp.h>

#include "core/perf/include/perf.hpp"
#include "omp/smirnov_p_matrix_fox_omp/include/ops_omp.hpp"

using namespace SmirnovOMP;

TEST(fox_blocked_omp, test_pipeline_run) {
  // Create data
  const int rows = 512;
  const int cols = 512;
  std::vector<double> matrix_a(rows * cols);
  std::vector<double> matrix_b(rows * cols);
  std::vector<double> matrix_c(rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_a[i * cols + j] = i + j;
      matrix_b[i * cols + j] = i - j;
      matrix_c[i * cols + j] = 0;
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(cols);
  // Create Task
  auto testTaskOMP = std::make_shared<FoxBlockedParallel>(taskDataSeq);
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  // Compare results
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double expected = 0;
      for (int k = 0; k < cols; ++k) {
        expected += matrix_a[i * cols + k] * matrix_b[k * cols + j];
      }
      ASSERT_EQ(matrix_c[i * cols + j], expected);
    }
  }
}
TEST(fox_blocked_omp, test_task_run) {
  // Create data
  const int rows = 512;
  const int cols = 512;
  std::vector<double> matrix_a(rows * cols);
  std::vector<double> matrix_b(rows * cols);
  std::vector<double> matrix_c(rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_a[i * cols + j] = i + j;
      matrix_b[i * cols + j] = i - j;
      matrix_c[i * cols + j] = 0;
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(cols);
  // Create Task
  auto testTaskOMP = std::make_shared<FoxBlockedParallel>(taskDataSeq);
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  // Compare results
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double expected = 0;
      for (int k = 0; k < cols; ++k) {
        expected += matrix_a[i * cols + k] * matrix_b[k * cols + j];
      }
      ASSERT_EQ(matrix_c[i * cols + j], expected);
    }
  }
}