#include <stan/math/prim.hpp>
#include <test/unit/math/util.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, one_hot_row_vector) {
  for (int K = 1; K < 5; K++) {
    for (int k = 1; k <= K; k++) {
      Eigen::RowVectorXd y = Eigen::RowVectorXd::Zero(K);
      y[k - 1] = 1;
      stan::test::expect_matrix_eq(y, stan::math::one_hot_row_vector(K, k));
    }
  }
}

TEST(MathFunctions, one_hot_row_vector_throw) {
  using stan::math::one_hot_row_vector;
  int K = 5;
  int k = 2;

  EXPECT_THROW(one_hot_row_vector(-1, k), std::domain_error);
  EXPECT_THROW(one_hot_row_vector(0, k), std::domain_error);
  EXPECT_THROW(one_hot_row_vector(K, K + 1), std::domain_error);
  EXPECT_THROW(one_hot_row_vector(K, 0), std::domain_error);
  EXPECT_THROW(one_hot_row_vector(K, -1), std::domain_error);
}
