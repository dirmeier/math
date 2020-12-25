#include <limits>
#include <vector>
#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/tools/promotion.hpp>

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/math/prim/fun/sign.hpp>
#include <stan/math/prim/fun/fabs.hpp>
#include <test/unit/math/prim/prob/vector_rng_test_helper.hpp>
#include <test/unit/math/prim/prob/util.hpp>

template <typename T1, typename T2, typename T3, typename T4>
inline typename boost::math::tools::promote_args<T1, T2, T3, T4>::type
skew_de_test(const T1& y, const T2& mu, const T3& sigma, const T4& tau) {
  using std::log;

  return log(2) + log(tau) + log(1 - tau) - log(sigma)
         - 2 * ((y < mu) ? (1 - tau) * (mu - y) : tau * (y - mu)) / sigma;
}

TEST(ProbDistributionsSkewedDoubleExponential,
     lpdf_computes_correct_gradients) {
  using stan::math::skew_double_exponential_lpdf;

  for (double ys : {0.2, 0.9, 1.1, 3.2}) {
    for (double sigmas : {0.1, 1.1, 3.2}) {
      for (double taus : {0.01, 0.1, 0.5, 0.9, 0.99}) {
        stan::math::var y = ys;
        stan::math::var mu = 1.0;
        stan::math::var sigma = sigmas;
        stan::math::var tau = taus;

        stan::math::var lp = skew_double_exponential_lpdf(y, mu, sigma, tau);
        std::vector<stan::math::var> theta;
        theta.push_back(y);
        theta.push_back(mu);
        theta.push_back(sigma);
        theta.push_back(tau);
        std::vector<double> grads;
        lp.grad(theta, grads);

        stan::math::var y_true = ys;
        stan::math::var mu_true = 1.0;
        stan::math::var sigma_true = sigmas;
        stan::math::var tau_true = taus;

        stan::math::var lp_test
            = skew_de_test(y_true, mu_true, sigma_true, tau_true);
        std::vector<stan::math::var> theta_true;
        theta_true.push_back(y_true);
        theta_true.push_back(mu_true);
        theta_true.push_back(sigma_true);
        theta_true.push_back(tau_true);
        std::vector<double> grads_true;
        lp_test.grad(theta_true, grads_true);

        EXPECT_NEAR(grads_true[0], grads[0], 0.001);
        EXPECT_NEAR(grads_true[1], grads[1], 0.001);
        EXPECT_NEAR(grads_true[2], grads[2], 0.001);
        EXPECT_NEAR(grads_true[3], grads[3], 0.001);
      }
    }
  }
}

TEST(ProbDistributionsPoissonBinomial, lpdf_works_on_scalar_arguments) {
  using stan::math::skew_double_exponential_lpdf;

  for (double ys : {0.2, 0.9, 1.1, 3.2}) {
    for (double mus : {0.1, 1.3, 3.0}) {
      for (double sigmas : {0.1, 1.1, 3.2}) {
        for (double taus : {0.01, 0.1, 0.5, 0.9, 0.99}) {
          EXPECT_NEAR(skew_de_test(ys, mus, sigmas, taus),
                      skew_double_exponential_lpdf(ys, mus, sigmas, taus),
                      0.001);
        }
      }
    }
  }
}

/*
 * We test the skew double exponential rng by setting tau=0.5 which recovers
 * a conventional double exponential distribution from Boost
 */
TEST(ProbDistributionsDoubleExponential, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = stan::math::round(2 * std::pow(N, 0.4));

  std::vector<double> samples;
  for (int i = 0; i < N; ++i) {
    samples.push_back(stan::math::skew_double_exponential_rng(2.0, 1.0, 0.5, rng));
  }

  // Generate quantiles from boost's double exponential distribution
  boost::math::laplace_distribution<> dist(2.0, 1.0);
  std::vector<double> quantiles;
  for (int i = 1; i < K; ++i) {
  double frac = static_cast<double>(i) / K;
  quantiles.push_back(quantile(dist, frac));
  }
  quantiles.push_back(std::numeric_limits<double>::max());

  // Assert that they match
  assert_matches_quantiles(samples, quantiles, 1e-6);
}
