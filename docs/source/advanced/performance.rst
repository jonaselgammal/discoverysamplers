Performance Optimization
========================

This guide covers techniques for optimizing the performance of ``discoverysamplers`` 
and how to write efficient likelihood functions.

Discovery's Built-in Optimizations
----------------------------------

``discoverysamplers`` is designed to interface with Discovery models, which are 
already optimized for performance using JAX. When you use Discovery likelihoods, 
many optimizations are handled automatically.

**Built-in optimizations in Discovery:**

- **JAX-based computation**: Discovery likelihoods use JAX for array operations
- **JIT compilation**: Likelihoods can be JIT-compiled for speed
- **Efficient signal computation**: Waveform templates are optimized
- **Precomputed quantities**: Noise covariances and FFTs are cached

**Enabling optimizations when initializing a sampler:**

.. code-block:: python

   from discoverysamplers import DiscoveryNessaiBridge, DiscoveryJAXNSBridge
   
   # For Nessai: enable JIT compilation
   bridge = DiscoveryNessaiBridge(discovery_model, priors, jit=True)
   
   # For JAX-NS: JIT is enabled by default, configure vectorization
   bridge = DiscoveryJAXNSBridge(discovery_model, priors, jit=True)
   bridge.configure_array_api(order=bridge.sampled_names)

   # For Eryn: JIT the likelihood before passing
   from jax import jit
   jitted_model = jit(discovery_model)
   bridge = DiscoveryErynBridge(jitted_model, priors)

**GPU acceleration:**

If you have a GPU available, JAX will automatically use it:

.. code-block:: python

   import jax
   
   # Check available devices
   print(f"Devices: {jax.devices()}")
   
   # Force GPU usage
   jax.config.update('jax_platform_name', 'gpu')
   
   # Enable 64-bit precision (recommended for nested sampling)
   jax.config.update("jax_enable_x64", True)

How to Best Write Your Own Likelihood
-------------------------------------

If you're extending Discovery or writing custom likelihoods, follow these 
guidelines for optimal performance.

Use JAX Instead of NumPy
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit

   # Slow: NumPy version
   def numpy_likelihood(params):
       x = params['x']
       y = params['y']
       return -0.5 * (np.square(x) + np.square(y))

   # Fast: JAX version
   @jit
   def jax_likelihood(params):
       x = params['x']
       y = params['y']
       return -0.5 * (jnp.square(x) + jnp.square(y))

Avoid Python Control Flow in JIT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Bad: Python if statement
   @jit
   def bad_likelihood(params):
       if params['x'] > 0:  # Python control flow
           return compute_A(params)
       else:
           return compute_B(params)

   # Good: JAX control flow
   @jit
   def good_likelihood(params):
       return jnp.where(
           params['x'] > 0,
           compute_A(params),
           compute_B(params)
       )

Precompute Fixed Quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class OptimizedLikelihood:
       def __init__(self, data):
           # Precompute once at initialization
           self.data = jnp.array(data)
           self.data_fft = jnp.fft.fft(data)
           self.noise_cov_inv = jnp.linalg.inv(compute_noise_cov(data))

       def __call__(self, params):
           # Use precomputed quantities
           signal = self.compute_signal(params)
           residual = self.data - signal
           return -0.5 * residual @ self.noise_cov_inv @ residual

Avoid Expensive Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Slow: Matrix inverse at every call
   def slow_likelihood(params):
       cov = build_covariance(params)
       inv_cov = jnp.linalg.inv(cov)  # Expensive!
       return -0.5 * residuals @ inv_cov @ residuals

   # Fast: Use solve instead
   def fast_likelihood(params):
       cov = build_covariance(params)
       return -0.5 * residuals @ jnp.linalg.solve(cov, residuals)

   # Faster: Cholesky decomposition
   def faster_likelihood(params):
       cov = build_covariance(params)
       L = jnp.linalg.cholesky(cov)
       y = jnp.linalg.solve(L, residuals)
       return -0.5 * jnp.sum(y**2) - jnp.sum(jnp.log(jnp.diag(L)))

Use Numerically Stable Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from jax.scipy.special import logsumexp

   # Unstable: numerical underflow
   def unstable(log_likes):
       likes = jnp.exp(log_likes)
       return jnp.log(jnp.sum(likes))

   # Stable: logsumexp
   def stable(log_likes):
       return logsumexp(log_likes)

Vectorize When Possible
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from jax import vmap

   # Scalar likelihood (evaluated one sample at a time)
   def scalar_likelihood(params):
       x = params['x']
       return -0.5 * x**2

   # Vectorized (evaluated on batches)
   @vmap
   def vectorized_likelihood(params):
       return scalar_likelihood(params)

Sampler-Specific Optimization
-----------------------------

Nessai
^^^^^^

**Flow configuration** for different problem complexities:

.. code-block:: python

   # Simple problems: smaller flow
   results = bridge.run_sampler(
       nlive=500,
       flow_config={
           'model_config': {
               'n_blocks': 4,
               'n_neurons': 32,
           }
       }
   )

   # Complex problems: larger flow
   results = bridge.run_sampler(
       nlive=1000,
       flow_config={
           'model_config': {
               'n_blocks': 8,
               'n_neurons': 64,
           }
       }
   )

**Multi-threading:**

.. code-block:: python

   results = bridge.run_sampler(
       nlive=1000,
       pytorch_threads=4,
   )

JAX-NS
^^^^^^

**Always enable vectorization** for best performance:

.. code-block:: python

   bridge = DiscoveryJAXNSBridge(model, priors, jit=True)
   bridge.configure_array_api(order=bridge.sampled_names)

Eryn
^^^^

**Walker count**: Use 2-4 times the number of parameters:

.. code-block:: python

   nwalkers = 4 * bridge.ndim
   bridge.create_sampler(nwalkers=nwalkers)

**Parallel likelihood evaluation:**

.. code-block:: python

   from multiprocessing import Pool

   with Pool(4) as pool:
       sampler = bridge.create_sampler(nwalkers=32, pool=pool)
       bridge.run_sampler(nsteps=10000)

Monitoring Performance
----------------------

Timing Your Computations
^^^^^^^^^^^^^^^^^^^^^^^^

**Timing the full sampling run:**

.. code-block:: python

   import time

   start = time.time()
   results = bridge.run_sampler(nlive=1000, output='output/')
   end = time.time()

   runtime = end - start
   n_likelihood_calls = results.get('total_likelihood_evaluations', 0)
   time_per_call = runtime / n_likelihood_calls if n_likelihood_calls > 0 else 0

   print(f"Total runtime: {runtime:.1f} s")
   print(f"Likelihood calls: {n_likelihood_calls}")
   print(f"Time per call: {time_per_call*1000:.3f} ms")

**Timing individual likelihood evaluations:**

.. code-block:: python

   import time
   import numpy as np

   # Generate test parameters
   test_params = {'x': 1.0, 'y': 2.0}

   # Warm-up call (important for JIT-compiled functions!)
   # The first call triggers compilation and will be much slower
   _ = model(test_params)

   # Time multiple evaluations
   n_calls = 1000
   start = time.time()
   for _ in range(n_calls):
       _ = model(test_params)
   end = time.time()

   time_per_call = (end - start) / n_calls
   print(f"Time per likelihood call: {time_per_call*1000:.3f} ms")
   print(f"Calls per second: {1/time_per_call:.1f}")

.. note::

   If you are using JIT compilation, the first evaluation triggers compilation 
   and will be significantly slower (often 10-100x). Always run a warm-up call 
   outside your timing loop, or discard the first timing measurement.

Profiling
^^^^^^^^^

**Basic Python profiling:**

.. code-block:: python

   import cProfile
   import pstats

   def run_sampling():
       bridge.run_sampler(nlive=100, max_samples=1000)

   # Profile sampling
   cProfile.run('run_sampling()', 'profile_stats')

   # Analyze results
   p = pstats.Stats('profile_stats')
   p.sort_stats('cumulative')
   p.print_stats(20)  # Top 20 slowest functions

**Memory profiling:**

.. code-block:: python

   import tracemalloc

   tracemalloc.start()
   results = bridge.run_sampler(nlive=1000, max_samples=10000)
   current, peak = tracemalloc.get_traced_memory()
   print(f"Peak memory: {peak / 1024**2:.1f} MB")
   tracemalloc.stop()

**JAX profiling with Perfetto:**

For detailed profiling of JAX computations (GPU/TPU activity, XLA operations, 
memory usage), use JAX's built-in profiler with `Perfetto <https://ui.perfetto.dev>`_:

.. code-block:: python

   import jax

   # Option 1: Context manager with automatic Perfetto link
   with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
       # Run the operations to be profiled
       results = bridge.run_sampler(nlive=100, max_samples=1000)
       # Block until computation is complete (important for async dispatch)
       if hasattr(results, 'block_until_ready'):
           results.block_until_ready()

   # Option 2: Manual start/stop for more control
   jax.profiler.start_trace("/tmp/jax-trace")
   results = bridge.run_sampler(nlive=100, max_samples=1000)
   jax.profiler.stop_trace()

After running, open the generated link or go to `ui.perfetto.dev <https://ui.perfetto.dev>`_ 
and load the trace file. The Perfetto UI provides:

- Timeline visualization of GPU/TPU operations
- Memory allocation tracking
- XLA operation breakdown
- Identification of performance bottlenecks

**Using XProf (TensorBoard profiling):**

For more advanced analysis, install XProf:

.. code-block:: bash

   pip install xprof

Then capture and view traces:

.. code-block:: python

   import jax

   # Start profiler server
   jax.profiler.start_server(9999)

   # Run your computation
   results = bridge.run_sampler(nlive=1000)

   # Stop when done
   jax.profiler.stop_server()

Launch the viewer:

.. code-block:: bash

   xprof --port 8791 /tmp/jax-trace

Navigate to ``http://localhost:8791/`` to view the trace. Use the "trace_viewer" 
tool to see a timeline of execution. See the 
`JAX profiling documentation <https://jax.readthedocs.io/en/latest/profiling.html>`_ 
for more details.

Troubleshooting Performance Issues
-----------------------------------

Likelihood Too Slow
^^^^^^^^^^^^^^^^^^^

1. Profile to find bottleneck
2. Convert to JAX + JIT
3. Precompute fixed quantities
4. Vectorize if possible
5. Use GPU if available

Out of Memory
^^^^^^^^^^^^^

1. Reduce ``nlive`` (nested sampling)
2. Reduce batch size (vectorization)
3. Use CPU instead of GPU
4. Stream data instead of loading all
5. Use float32 instead of float64

Sampler Not Converging
^^^^^^^^^^^^^^^^^^^^^^^

1. Check likelihood for bugs
2. Verify prior bounds
3. Increase ``nlive`` or walkers
4. Use parallel tempering (Eryn)
5. Simplify model if possible

See Also
--------

- :doc:`../user_guide/model_requirements` - Model implementation guidelines
- :doc:`../user_guide/nessai_usage` - Nessai-specific tips
- :doc:`../user_guide/jaxns_usage` - JAX-NS optimization
- `JAX documentation <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_ - Thinking in JAX
