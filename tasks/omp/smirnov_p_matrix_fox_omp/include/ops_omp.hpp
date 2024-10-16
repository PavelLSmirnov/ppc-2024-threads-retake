//  Copyright 2024 Smirnov Pavel

#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace SmirnovOMP {

using Matrix = std::vector<std::vector<double>>;

class FoxBlockedSequential : public ppc::core::Task {
 public:
  explicit FoxBlockedSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix A{};
  Matrix B{};
  Matrix C{};
  int block_size{};
};
class FoxBlockedParallel : public ppc::core::Task {
 public:
  explicit FoxBlockedParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix A{};
  Matrix B{};
  Matrix C{};
  int block_size{};
};
}  // namespace SmirnovOMP
