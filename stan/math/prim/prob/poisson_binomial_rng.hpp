#ifndef STAN_MATH_PRIM_PROB_POISSON_BINOMIAL_RNG_HPP
#define STAN_MATH_PRIM_PROB_POISSON_BINOMIAL_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <iostream>
namespace stan {
namespace math {

/** \ingroup prob_dists
 * Return a pseudorandom Poisson binomial random variable for the given vector
 * of success parameters using the specified random number
 * generator.
 *
 * @tparam RNG class of rng
 * @param theta (Sequence of) chance of success parameter(s)
 * @param rng random number generator
 * @return a Poisson binomial distribution random variable
 * @throw std::domain_error if theta is not a valid probability
 */
template <typename T_theta, typename RNG>
inline typename VectorBuilder<true, int, T_theta>::type
poisson_binomial_rng(const T_theta& theta, RNG& rng) {
  static const char* function = "poisson_binomial_rng";

  vector_seq_view<T_theta> theta_vec(theta);
  size_t N = size_mvt(theta);

  for (size_t i = 0; i < N; i++) {
    check_finite(function, "Probability parameters", theta_vec[i]);
    check_bounded(function, "Probability parameters", value_of(theta_vec[i]), 0.0, 1.0);
  }

  VectorBuilder<true, int, T_theta> output(N);
  for (size_t n = 0; n < N; ++n) {
    int y = 0;
    for (size_t i = 0; i < theta_vec[i].size(); ++i) {
      boost::variate_generator<RNG&, boost::bernoulli_distribution<> >
          bernoulli_rng(rng, boost::bernoulli_distribution<>(theta_vec[n][i]));
      y += bernoulli_rng();
    }
    output[n] = y;
  }

  return output.data();
}

}  // namespace math
}  // namespace stan
#endif
