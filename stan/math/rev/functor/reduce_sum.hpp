#ifndef STAN_MATH_REV_SCAL_FUNCTOR_REDUCE_SUM_HPP
#define STAN_MATH_REV_SCAL_FUNCTOR_REDUCE_SUM_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/rev/core/deep_copy_vars.hpp>
#include <stan/math/rev/core/accumulate_adjoints.hpp>
#include <stan/math/rev/core/count_vars.hpp>
#include <stan/math/rev/core/save_varis.hpp>
#include <stan/math/rev/core.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <tuple>
#include <vector>

namespace stan {
namespace math {
namespace internal {

/**
 * Var specialization of reduce_sum_impl
 *
 * @tparam ReduceFunction Type of reducer function
 * @tparam ReturnType Must be var
 * @tparam Vec Type of sliced argument
 * @tparam Args Types of shared arguments
 */
template <typename ReduceFunction, typename ReturnType, typename Vec,
          typename... Args>
struct reduce_sum_impl<ReduceFunction, require_var_t<ReturnType>, ReturnType,
                       Vec, Args...> {
  /**
   * Internal object meeting the Imperative form requirements of `tbb::parallel_reduce`
   *
   * @note see link [here](https://tinyurl.com/vp7xw2t) for requirements.
   */
  struct recursive_reducer {
    size_t per_job_sliced_terms_;
    size_t num_shared_terms_;  // Number of terms shared across threads
    double* sliced_partials_;  // Points to adjoints of the partial calculations
    Vec vmapped_;
    std::ostream* msgs_;
    std::tuple<Args...> args_tuple_;
    double sum_{0.0};
    Eigen::VectorXd args_adjoints_{0};

    template <typename VecT, typename... ArgsT>
    recursive_reducer(size_t per_job_sliced_terms, size_t num_shared_terms,
                      double* sliced_partials, VecT&& vmapped,
                      std::ostream* msgs, ArgsT&&... args)
        : per_job_sliced_terms_(per_job_sliced_terms),
          num_shared_terms_(num_shared_terms),
          sliced_partials_(sliced_partials),
          vmapped_(std::forward<VecT>(vmapped)),
          msgs_(msgs),
          args_tuple_(std::forward<ArgsT>(args)...) {}

    /*
     * This is the copy operator as required for tbb::parallel_reduce
     *   Imperative form. This requires the reduced values (sum_ and arg_adjoints_)
     *   be reset to zero.
     */
    recursive_reducer(recursive_reducer& other, tbb::split)
        : per_job_sliced_terms_(other.per_job_sliced_terms_),
          num_shared_terms_(other.num_shared_terms_),
          sliced_partials_(other.sliced_partials_),
          vmapped_(other.vmapped_),
          msgs_(other.msgs_),
          args_tuple_(other.args_tuple_) {}

    /**
     * Compute, using nested autodiff, the value and Jacobian of `ReduceFunction`
     *   called over the range defined by r and accumulate those in member
     *   variable sum_ (for the value) and args_adjoints_ (for the Jacobian).
     *   This function may be called multiple times per object instantiation
     *   (so the sum_ and args_adjoints_ must be accumulated, not just assigned).
     *
     * @param r Range over which to compute reduce_sum
     */
    inline void operator()(const tbb::blocked_range<size_t>& r) {
      if (r.empty()) {
        return;
      }

      if (args_adjoints_.size() == 0) {
        args_adjoints_ = Eigen::VectorXd::Zero(this->num_shared_terms_);
      }

      // Initialize nested autodiff stack
      const nested_rev_autodiff begin_nest;

      // Create nested autodiff copies of sliced argument that do not point
      //   back to main autodiff stack
      std::decay_t<Vec> local_sub_slice;
      local_sub_slice.reserve(r.size());
      for (int i = r.begin(); i < r.end(); ++i) {
        local_sub_slice.emplace_back(deep_copy_vars(vmapped_[i]));
      }

      // Create nested autodiff copies of all shared arguments that do not point
      //   back to main autodiff stack
      auto args_tuple_local_copy = apply(
          [&](auto&&... args) {
            return std::tuple<decltype(deep_copy_vars(args))...>(deep_copy_vars(args)...);
          },
          this->args_tuple_);

      // Perform calculation
      var sub_sum_v = apply(
          [&](auto&&... args) {
            return ReduceFunction()(r.begin(), r.end() - 1, local_sub_slice,
                                    this->msgs_, args...);
          },
          args_tuple_local_copy);

      // Compute Jacobian
      sub_sum_v.grad();

      // Accumulate value of reduce_sum
      sum_ += sub_sum_v.val();

      // Accumulate adjoints of sliced_arguments
      accumulate_adjoints(
          this->sliced_partials_ + r.begin() * per_job_sliced_terms_,
          local_sub_slice);

      // Accumulate adjoints of shared_arguments
      apply(
          [&](auto&&... args) {
            accumulate_adjoints(args_adjoints_.data(),
				std::forward<decltype(args)>(args)...);
          },
          std::move(args_tuple_local_copy));
    }

    /**
     * Join reducers. Accumuluate the value (sum_) and Jacobian (arg_adoints_)
     *   of the other reducer.
     *
     * @param rhs Another partial sum
     */
    void join(const recursive_reducer& rhs) {
      this->sum_ += rhs.sum_;
      if (this->args_adjoints_.size() != 0 && rhs.args_adjoints_.size() != 0) {
        this->args_adjoints_ += rhs.args_adjoints_;
      } else if (this->args_adjoints_.size() == 0
                 && rhs.args_adjoints_.size() != 0) {
        this->args_adjoints_ = rhs.args_adjoints_;
      }
    }
  };

  /**
   * Call an instance of the function `ReduceFunction` on every element
   *   of an input sequence and sum these terms.
   *
   * This specialization is parallelized using tbb and works for reverse
   *   mode autodiff.
   *
   * An instance, f, of `ReduceFunction` should have the signature:
   *   var f(int start, int end, Vec&& vmapped_subset, std::ostream* msgs, Args&&... args)
   *
   * `ReduceFunction` must be default constructible without any arguments
   *
   * Each call to `ReduceFunction` is responsible for computing the
   *   start through end - 1 terms of the overall sum. All args are passed
   *   from this function through to the `ReduceFunction` instances.
   *   However, only elements start through end - 1 of the vmapped argument are
   *   passed to the `ReduceFunction` instances (as the `vmapped_subset` argument).
   *
   * This function distributes computation of the desired sum and the Jacobian of
   *   that sum over multiple threads by coordinating calls to `ReduceFunction` instances.
   *   Results are stored as precomputed varis in the autodiff tree.
   *
   * @param vmapped Sliced arguments used only in some sum terms
   * @param grainsize Suggested grainsize for tbb
   * @param[in, out] msgs The print stream for warning messages
   * @param args Shared arguments used in every sum term
   * @return Summation of all terms
   */
  inline var operator()(Vec&& vmapped, std::size_t grainsize,
                        std::ostream* msgs, Args&&... args) const {
    const std::size_t num_jobs = vmapped.size();

    if (num_jobs == 0) {
      return var(0.0);
    }

    const std::size_t per_job_sliced_terms = count_vars(vmapped[0]);
    const std::size_t num_sliced_terms = num_jobs * per_job_sliced_terms;
    const std::size_t num_shared_terms = count_vars(args...);

    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
        num_sliced_terms + num_shared_terms);
    double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(
        num_sliced_terms + num_shared_terms);

    for (size_t i = 0; i < num_sliced_terms; ++i) {
      partials[i] = 0.0;
    }
    recursive_reducer worker(per_job_sliced_terms, num_shared_terms, partials,
                             vmapped, msgs, args...);

#ifdef STAN_DETERMINISTIC
    tbb::inline_partitioner partitioner;
    tbb::parallel_deterministic_reduce(
        tbb::blocked_range<std::size_t>(0, num_jobs, grainsize), worker,
        partitioner);
#else
    tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, num_jobs, grainsize), worker);
#endif

    save_varis(varis, std::forward<Vec>(vmapped));
    save_varis(varis + num_sliced_terms, std::forward<Args>(args)...);

    for (size_t i = 0; i < num_shared_terms; ++i) {
      partials[num_sliced_terms + i] = worker.args_adjoints_(i);
    }

    return var(new precomputed_gradients_vari(
        worker.sum_, num_sliced_terms + num_shared_terms, varis, partials));
  }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

#endif
