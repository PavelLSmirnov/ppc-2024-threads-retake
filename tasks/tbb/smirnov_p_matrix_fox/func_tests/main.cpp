// Copyright 2024 Smirnov Pavel
#include <gtest/gtest.h>
#include <tbb/tbb.h>

#include <iostream>
#include <vector>

#include "tbb/smirnov_p_matrix_fox/include/matrix_fox.hpp"

TEST(Smirnov_Pavel_Mult_Matrix_Fox, validationTest) {
  size_t n = 2;
  std::vector<double> in1{2.0, 5.0, 3.0, -1.0};
  std::vector<double> in2{1.0, 0.0, 0.0, 1.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MatrixFoxAlgTaskTBB taskTbb(taskDataSeq);
  ASSERT_EQ(taskTbb.validation(), true);
}

TEST(Smirnov_Pavel_Mult_Matrix_Fox, mult2) {
  size_t n = 10;
  std::vector<double> in1(n * n, 10.0);
  std::vector<double> in2(n * n, 1.0);
  std::vector<double> out(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MatrixFoxAlgTaskTBB taskTbb(taskDataSeq);
  ASSERT_EQ(taskTbb.validation(), true);
  ASSERT_TRUE(taskTbb.pre_processing());
  ASSERT_TRUE(taskTbb.run());
  ASSERT_TRUE(taskTbb.post_processing());

  std::vector<double> expected_result = mulMatrix(in1, in2, n);

  double epsilon = 1e-6;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_NEAR(out[i * n + j], expected_result[i * n + j], epsilon);
    }
  }
}

TEST(Smirnov_Pavel_Mult_Matrix_Fox, mult3) {
  size_t n = 10;
  std::vector<double> in1(n * n, 10.0);
  std::vector<double> in2(n * n, 10.0);
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MatrixFoxAlgTaskTBB taskTbb(taskDataSeq);
  ASSERT_EQ(taskTbb.validation(), true);
  ASSERT_TRUE(taskTbb.pre_processing());
  ASSERT_TRUE(taskTbb.run());
  ASSERT_TRUE(taskTbb.post_processing());

  std::vector<double> expected_result = mulMatrix(in1, in2, n);

  double epsilon = 1e-6;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_NEAR(out[i * n + j], expected_result[i * n + j], epsilon);
    }
  }
}

TEST(Smirnov_Pavel_Mult_Matrix_Fox, mult4) {
  size_t n = 4;
  std::vector<double> in1{1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0};
  std::vector<double> in2{2.0,     6.0,     18.0,     54.0,     162.0,     486.0,     1458.0,    4374.0,
                          13122.0, 39366.0, 118098.0, 354294.0, 1062882.0, 3188646.0, 9565938.0, 28697814.0};
  std::vector<double> out(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MatrixFoxAlgTaskTBB TaskTbb(taskDataSeq);
  ASSERT_EQ(TaskTbb.validation(), true);
  ASSERT_TRUE(TaskTbb.pre_processing());
  ASSERT_TRUE(TaskTbb.run());
  ASSERT_TRUE(TaskTbb.post_processing());

  std::vector<double> expected_result = mulMatrix(in1, in2, n);

  double epsilon = 1e-6;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_NEAR(out[i * n + j], expected_result[i * n + j], epsilon);
    }
  }
}

TEST(Smirnov_Pavel_Mult_Matrix_Fox, mult5) {
  size_t n = 4;
  std::vector<double> in1{1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0};
  std::vector<double> in2{10.0, 9.0, 8.0, 7.0, 9.0, 8.0, 7.0, 6.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0};
  std::vector<double> out(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  MatrixFoxAlgTaskTBB TaskTbb(taskDataSeq);
  ASSERT_EQ(TaskTbb.validation(), true);
  ASSERT_TRUE(TaskTbb.pre_processing());
  ASSERT_TRUE(TaskTbb.run());
  ASSERT_TRUE(TaskTbb.post_processing());

  std::vector<double> expected_result = mulMatrix(in1, in2, n);

  double epsilon = 1e-6;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_NEAR(out[i * n + j], expected_result[i * n + j], epsilon);
    }
  }
}
