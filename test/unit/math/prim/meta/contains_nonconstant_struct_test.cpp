
#include <stan/math/prim/meta.hpp>
#include <gtest/gtest.h>

#include <vector>





typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> const_t1;
typedef std::vector<const_t1> const_t2;
typedef std::vector<const_t2> const_t3;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> const_u1;
typedef std::vector<const_u1> const_u2;
typedef std::vector<const_u2> const_u3;

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> const_v1;
typedef std::vector<const_v1> const_v2;
typedef std::vector<const_v2> const_v3;

TEST(MetaTraits_mat, containsNonconstantStruct) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::contains_nonconstant_struct;

  EXPECT_FALSE(contains_nonconstant_struct<const_t1>::value);
  EXPECT_FALSE(contains_nonconstant_struct<const_t3>::value);
  EXPECT_FALSE(contains_nonconstant_struct<const_u1>::value);
  EXPECT_FALSE(contains_nonconstant_struct<const_u3>::value);
  EXPECT_FALSE(contains_nonconstant_struct<const_v2>::value);
}
