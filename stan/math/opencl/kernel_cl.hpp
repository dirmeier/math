#ifndef STAN_MATH_OPENCL_KERNEL_CL_HPP
#define STAN_MATH_OPENCL_KERNEL_CL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/buffer_types.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/stringify.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/opencl/kernels/helpers.hpp>
#include <stan/math/prim/fun/vec_concat.hpp>
#include <CL/cl2.hpp>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <utility>

namespace stan {
namespace math {
namespace opencl_kernels {
namespace internal {
/** \ingroup kernel_executor_opencl
 * Extracts the kernel's arguments, used in the global and local kernel
 * constructor.
 * @tparam For this general template the function will return back the
 * value passed in.
 * @param t The type that will be returned.
 * @return the input t.
 */
template <typename T, require_not_matrix_cl_t<T>* = nullptr>
inline const T& get_kernel_args(const T& t) {
  return t;
}

/** \ingroup kernel_executor_opencl
 * Extracts the kernel's arguments, used in the global and local kernel
 * constructor.
 * @tparam K The type of the \c matrix_cl.
 * @param m The \c matrix with an OpenCL Buffer.
 * @return The OpenCL Buffer.
 */
template <typename K, require_matrix_cl_t<K>* = nullptr>
inline const cl::Buffer& get_kernel_args(const K& m) {
  return m.buffer();
}

/** \ingroup kernel_executor_opencl
 * Helper function for assigning events to a \c matrix_cl.
 *
 * @tparam T Whether the assignment is to an \c in_buffer, \c out_buffer, or \c
 * in_out_buffer.
 * @tparam K The type of the \c matrix_cl.
 *
 */
template <typename T, typename K = double>
struct assign_event_helper {
  /** \ingroup kernel_executor_opencl
   * Assigns the event to the \c matrix_cl.
   * @param e the event to be assigned.
   * @param m The \c matrix_cl to be assigned to.
   */
  inline void set(const cl::Event& e, const K& m) {}
};

// Specialization for \c in_buffer
template <typename K>
struct assign_event_helper<in_buffer, K> {
  inline void set(const cl::Event& e, const K& m) { m.add_read_event(e); }
};

// Specialization for \c out_buffer
template <typename K>
struct assign_event_helper<out_buffer, K> {
  inline void set(const cl::Event& e, const K& m) { m.add_write_event(e); }
};

// Specialization for \c in_out_buffer
template <typename K>
struct assign_event_helper<in_out_buffer, K> {
  inline void set(const cl::Event& e, const K& m) { m.add_read_write_event(e); }
};

/** \ingroup kernel_executor_opencl
 * Assigns the event to a \c matrix_cl.
 * @tparam T The type to be assigned, if not a matrix_cl this function
 * will do nothing.
 * @tparam K The type of the \c matrix_cl.
 * @param e The event to be assigned.
 */
template <typename T, typename K = double>
inline void assign_event(const cl::Event& e, const T&) {}

/** \ingroup kernel_executor_opencl
 * Assigns the event to a \c matrix_cl
 * @tparam T The type to be assigned, if not a matrix_cl will do nothing.
 * @tparam K The type of the \c matrix_cl.
 * @param e The event to be assigned.
 * @param m The \c matrix_cl to be assigned
 */
template <typename T, typename K, require_matrix_cl_t<K>* = nullptr>
inline void assign_event(const cl::Event& e, const K& m) {
  assign_event_helper<T, K> helper;
  helper.set(e, m);
}

template <typename T, require_same_t<T, cl::Event>* = nullptr>
inline void assign_events(const T&) {}

/** \ingroup kernel_executor_opencl
 * Adds the event to any \c matrix_cls in the arguments depending on whether
 * they are \c in_buffer, \c out_buffer, or \c in_out_buffers.
 * @tparam Arg Arguments given during kernel creation that specify the kernel
 * signature.
 * @tparam Args Arguments given during kernel creation that specify the kernel
 * signature.
 * @tparam CallArg First argument type used to call the kernel
 * @tparam CallArgs Other argument types used to call the kernel.
 * @param new_event The cl::Event generated involving the arguments.
 * @param m Arguments to the kernel that may be \c matrix_cls or not.
 * Non-matrices are ignored.
 * @param args Arguments to the kernel that may be matrices or not.
 * Non-matrices are ignored.
 */
template <typename Arg, typename... Args, typename CallArg,
          typename... CallArgs>
inline void assign_events(const cl::Event& new_event, CallArg& m,
                          CallArgs&... args) {
  assign_event<Arg>(new_event, m);
  assign_events<Args...>(new_event, args...);
}

/** \ingroup kernel_executor_opencl
 * Helper function to select OpenCL event vectors from an \c matrix_cl
 * @tparam T For non \c matrix_cl types, the type of the first argument.
 * Otherwise this is the in/out/inout buffer type.
 * @tparam K For \c matrix_cl types, the type of the matrix_cl
 */
template <typename T, typename K = double>
struct select_event_helper {
  /** \ingroup kernel_executor_opencl
   * Get the events from a matrix_cl. For non \c matrix_cl types this will do
   * nothing.
   * @param m A type to extract the event from.
   */
  inline const std::vector<cl::Event> get(const T& m) {
    return std::vector<cl::Event>();
  }
};

// Specialization for in_buffer
template <typename K>
struct select_event_helper<in_buffer, K> {
  inline const std::vector<cl::Event> get(const K& m) {
    return m.write_events();
  }
};

// Specialization for out_buffer
template <typename K>
struct select_event_helper<out_buffer, K> {
  inline const std::vector<cl::Event> get(const K& m) {
    return m.read_write_events();
  }
};

// Specialization for in_out_buffer
template <typename K>
struct select_event_helper<in_out_buffer, K> {
  inline const std::vector<cl::Event> get(const K& m) {
    return m.read_write_events();
  }
};

/** \ingroup kernel_executor_opencl
 * Select events from kernel arguments. Does nothing for non \c matrix_cl types.
 * @tparam T The argument type for a non \c matrix_cl, else the in/out/in_out
 * buffer types.
 * @tparam K The type of the \c matrix_cl
 * @param m If an \c matrix_cl, gets the event vector, else this argument does
 * nothing.
 * @return A vector of OpenCL events.
 */
template <typename T, typename K = double>
inline const std::vector<cl::Event> select_events(const T& m) {
  select_event_helper<T, K> helper;
  return helper.get(m);
}

// Specialization for \c matrix_cl
template <typename T, typename K, require_matrix_cl_t<K>* = nullptr>
inline const std::vector<cl::Event> select_events(const K& m) {
  select_event_helper<T, K> helper;
  return helper.get(m);
}

}  // namespace internal

/** \ingroup kernel_executor_opencl
 * Compile an OpenCL kernel.
 *
 * @param name The name for the kernel
 * @param sources A std::vector of strings containing the code for the kernel.
 * @param options The values of macros to be passed at compile time.
 */
inline auto compile_kernel(const char* name,
                           const std::vector<std::string>& sources,
                           const std::map<std::string, int>& options) {
  auto base_opts = opencl_context.base_opts();
  for (auto& it : options) {
    if (base_opts[it.first] > it.second) {
      base_opts[it.first] = it.second;
    }
  }
  std::string kernel_opts = "";
  for (auto&& comp_opts : base_opts) {
    kernel_opts += std::string(" -D") + comp_opts.first + "="
                   + std::to_string(comp_opts.second);
  }
  cl::Program program(opencl_context.context(), sources);
  try {
    program.build({opencl_context.device()}, kernel_opts.c_str());

    return cl::Kernel(program, name);
  } catch (const cl::Error& e) {
    // in case of CL_BUILD_PROGRAM_FAILURE, print the build error
    if (e.err() == -11) {
      std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
          opencl_context.device()[0]);
      system_error("compile_kernel", name, e.err(), buildlog.c_str());
    } else {
      check_opencl_error(name, e);
    }
  }
  return cl::Kernel();  // never reached because check_opencl_error throws
}

/** \ingroup kernel_executor_opencl
 * Creates functor for kernels
 *
 * @tparam Args Parameter pack of all kernel argument types.
 */
template <typename... Args>
struct kernel_cl {
 private:
  const char* name_;
  std::vector<std::string> sources_;
  std::map<std::string, int> opts_;
  mutable cl::Kernel kernel_;

 public:
  /** \ingroup kernel_executor_opencl
   * Creates functor for kernels that only need access to defining
   *  the global work size.
   * @param name The name for the kernel
   * @param sources A std::vector of strings containing the code for the kernel.
   * @param options The values of macros to be passed at compile time.
   */
  kernel_cl(const char* name, std::vector<std::string> sources,
            std::map<std::string, int> options = {})
      : name_(name), sources_(std::move(sources)), opts_(std::move(options)) {}

  /** \ingroup kernel_executor_opencl
   * Executes a kernel
   * @tparam CallArgs The types of the callee arguments.
   * @tparam Args Parameter pack of all kernel argument types.
   * @param global_thread_size The global work size.
   * @param args The arguments to pass to the kernel.
   * @return An Opencl event.
   */
  template <typename... CallArgs>
  auto operator()(cl::NDRange global_thread_size,
                  const CallArgs&... args) const {
    if (kernel_() == NULL) {
      kernel_ = compile_kernel(name_, sources_, opts_);
      opencl_context.register_kernel_cache(&kernel_);
    }
    cl::EnqueueArgs eargs(opencl_context.queue(),
                          vec_concat(internal::select_events<Args>(args)...),
                          global_thread_size);
    cl::KernelFunctor<internal::to_const_buffer_t<Args>&...> kernel_functor(
        kernel_);
    cl::Event kern_event
        = kernel_functor(eargs, internal::get_kernel_args(args)...);
    internal::assign_events<Args...>(kern_event, args...);
    return kern_event;
  }

  /** \ingroup kernel_executor_opencl
   * Executes a kernel
   * @tparam CallArgs The types of the callee arguments.
   * @tparam Args Parameter pack of all kernel argument types.
   * @param global_thread_size The global work size.
   * @param thread_block_size The thread block size.
   * @param args The arguments to pass to the kernel.
   * @return An Opencl event.
   */
  template <typename... CallArgs>
  auto operator()(cl::NDRange global_thread_size, cl::NDRange thread_block_size,
                  const CallArgs&... args) const {
    if (kernel_() == NULL) {
      kernel_ = compile_kernel(name_, sources_, opts_);
      opencl_context.register_kernel_cache(&kernel_);
    }
    cl::EnqueueArgs eargs(opencl_context.queue(),
                          vec_concat(internal::select_events<Args>(args)...),
                          global_thread_size, thread_block_size);
    cl::KernelFunctor<internal::to_const_buffer_t<Args>&...> kernel_functor(
        kernel_);
    cl::Event kern_event
        = kernel_functor(eargs, internal::get_kernel_args(args)...);
    internal::assign_events<Args...>(kern_event, args...);
    return kern_event;
  }

  /** \ingroup kernel_executor_opencl
   * Retrieves an option used for compiling the kernel.
   * @param option_name which option to retrieve
   * @return option value
   */
  int get_option(const std::string option_name) const {
    return std::min(opts_.at(option_name),
                    opencl_context.base_opts().at(option_name));
  }
};

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan

#endif
#endif
