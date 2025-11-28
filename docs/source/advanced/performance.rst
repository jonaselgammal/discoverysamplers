Performance Optimization
========================

This guide covers techniques for optimizing the performance of ``discoverysamplers`` across different sampler backends.

General Optimization Strategies
--------------------------------

Profile Your Likelihood
^^^^^^^^^^^^^^^^^^^^^^^

Before optimizing, identify bottlenecks:

.. code-block:: python

   import time
   import numpy as np

   def profile_likelihood(model, bridge, n_calls=1000):
       """Profile likelihood function performance."""
       # Sample from priors
       test_params = bridge.sample_priors(n=n_calls)

       # Time likelihood evaluations
       start = time.time()
       for params in test_params:
           log_L = model(params)
       end = time.time()

       time_per_call = (end - start) / n_calls
       print(f"Time per likelihood call: {time_per_call*1000:.3f} ms")
       print(f"Calls per second: {1/time_per_call:.1f}")

       return time_per_call

   # Profile your model
   profile_likelihood(my_model, bridge)

Optimize Hotspots
^^^^^^^^^^^^^^^^^

Use Python profiling tools to find slow code:

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

JAX Optimization
----------------

Use JAX for Computation
^^^^^^^^^^^^^^^^^^^^^^^

JAX provides significant speedups:

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit

   # NumPy version (slow)
   def numpy_model(params):
       x = params['x']
       y = params['y']
       return -0.5 * (np.square(x) + np.square(y))

   # JAX version (fast)
   def jax_model(params):
       x = params['x']
       y = params['y']
       return -0.5 * (jnp.square(x) + jnp.square(y))

   # Enable JIT
   bridge = DiscoveryNessaiBridge(jax_model, priors, jit=True)

**Speedup**: 10-100x for complex models

JIT Compilation Best Practices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from jax import jit

   # Good: JIT the entire likelihood
   @jit
   def fast_likelihood(params):
       # All operations in JAX
       signal = compute_signal_jax(params)
       return compute_log_L_jax(signal)

   # Bad: JIT with side effects
   @jit
   def bad_likelihood(params):
       # Don't print inside JIT
       print(params)  # Will cause issues
       return log_L

   # Bad: JIT with Python control flow
   @jit
   def bad_likelihood2(params):
       if params['x'] > 0:  # Python if, not JAX
           return compute_A(params)
       else:
           return compute_B(params)

   # Good: Use JAX control flow
   @jit
   def good_likelihood2(params):
       return jnp.where(
           params['x'] > 0,
           compute_A(params),
           compute_B(params)
       )

Enable 64-bit Precision Carefully
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax

   # 64-bit: more accurate but slower
   jax.config.update("jax_enable_x64", True)

   # 32-bit: faster but less accurate
   jax.config.update("jax_enable_x64", False)

**Recommendation**: Use 64-bit for nested sampling, 32-bit may be OK for MCMC

Vectorization
^^^^^^^^^^^^^

Vectorize likelihood for massive speedups:

.. code-block:: python

   from jax import vmap

   # Scalar likelihood
   def scalar_likelihood(params):
       """Works with single parameter set."""
       x = params['x']  # Scalar
       return -0.5 * x**2

   # Vectorized version
   def vectorized_likelihood(params):
       """Works with batched parameters."""
       x = params['x']  # Shape: (N,)
       return -0.5 * x**2  # Shape: (N,)

   # Or use vmap to vectorize automatically
   @vmap
   def auto_vectorized(params):
       return scalar_likelihood(params)

   # Enable in JAX-NS
   bridge = DiscoveryJAXNSBridge(vectorized_likelihood, priors, jit=True)
   bridge.configure_array_api(order=['x', 'y'])

**Speedup**: 5-50x depending on batch size

GPU Acceleration
----------------

Using GPUs with JAX
^^^^^^^^^^^^^^^^^^^

JAX automatically uses GPU if available:

.. code-block:: python

   import jax

   # Check available devices
   print(f"Devices: {jax.devices()}")

   # Force GPU
   jax.config.update('jax_platform_name', 'gpu')

   # Create JAX model
   bridge = DiscoveryJAXNSBridge(jax_model, priors, jit=True)

   # Sampling automatically uses GPU
   results = bridge.run_sampler(nlive=1000, max_samples=10000)

**Speedup**: 10-1000x for large models

Memory Management on GPU
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Limit GPU memory growth
   import os
   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

   # Or set memory fraction
   os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

   # If out of memory, reduce batch size or use CPU
   jax.config.update('jax_platform_name', 'cpu')

Precomputation and Caching
---------------------------

Precompute Fixed Quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class OptimizedModel:
       def __init__(self, data):
           # Precompute expensive quantities once
           self.data = data
           self.data_fft = jnp.fft.fft(data)
           self.whitening_matrix = jnp.linalg.inv(jnp.cov(data))
           self.log_det = jnp.linalg.slogdet(jnp.cov(data))[1]

       def __call__(self, params):
           # Use precomputed quantities
           signal_fft = self.compute_signal_fft(params)
           residual_fft = self.data_fft - signal_fft

           # Fast whitened likelihood
           return self.fast_likelihood(residual_fft)

   model = OptimizedModel(data)
   bridge = DiscoveryNessaiBridge(model, priors, jit=True)

Cache Repeated Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from functools import lru_cache

   class CachedModel:
       def __init__(self):
           self._cache = {}

       def __call__(self, params):
           # Create hashable key
           key = tuple(sorted(params.items()))

           if key not in self._cache:
               self._cache[key] = self._compute_likelihood(params)

           return self._cache[key]

       def _compute_likelihood(self, params):
           # Expensive computation
           return log_likelihood

   # Warning: Only use if likelihood is deterministic
   # and parameter space is discrete/limited

Numerical Optimization
----------------------

Avoid Expensive Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Use Stable Algorithms
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from scipy.special import logsumexp

   # Unstable: numerical underflow
   def unstable(log_likes):
       likes = np.exp(log_likes)
       total = np.sum(likes)
       return np.log(total)

   # Stable: logsumexp
   def stable(log_likes):
       return logsumexp(log_likes)

   # In JAX
   from jax.scipy.special import logsumexp as jax_logsumexp

Reduce Precision Where Possible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use float32 for intermediate calculations if possible
   def mixed_precision_model(params):
       # Cast to float32 for computation
       x_32 = jnp.float32(params['x'])
       y_32 = jnp.float32(params['y'])

       # Compute in float32 (faster)
       result_32 = compute_expensive(x_32, y_32)

       # Cast back to float64 for final result
       return jnp.float64(result_32)

Sampler-Specific Optimization
------------------------------

Nessai
^^^^^^

**Flow Configuration**:

.. code-block:: python

   # Start with small flow for quick exploration
   results = bridge.run_sampler(
       nlive=500,
       output='output/quick/',
       flow_config={
           'model_config': {
               'n_blocks': 4,
               'n_neurons': 32,
           }
       }
   )

   # Then use larger flow for final run
   results = bridge.run_sampler(
       nlive=1000,
       output='output/final/',
       flow_config={
           'model_config': {
               'n_blocks': 8,
               'n_neurons': 64,
           }
       }
   )

**Proposal Efficiency**:

.. code-block:: python

   # Increase pool size for better proposals
   results = bridge.run_sampler(
       nlive=1000,
       n_pool=2000,  # 2x nlive
       poolsize_scale=10,
   )

**Multi-threading**:

.. code-block:: python

   # Enable PyTorch multi-threading
   results = bridge.run_sampler(
       nlive=1000,
       pytorch_threads=4,
   )

JAX-NS
^^^^^^

**Vectorization** (key for performance):

.. code-block:: python

   # Always use vectorization
   bridge.configure_array_api(order=bridge.sampled_names)

**Batch Size**:

.. code-block:: python

   # Adjust batch size for GPU memory
   bridge.configure_array_api(
       order=bridge.sampled_names,
       batch_size=100  # Tune based on GPU memory
   )

Eryn
^^^^

**Walker Count**:

.. code-block:: python

   # More walkers = better exploration but slower per step
   # Optimal: 2-4 times the number of parameters
   nwalkers = 4 * bridge.ndim

**Parallelization**:

.. code-block:: python

   from multiprocessing import Pool

   # Parallel likelihood evaluation
   with Pool(4) as pool:
       sampler = bridge.create_sampler(
           nwalkers=32,
           pool=pool
       )
       sampler.run_mcmc(initial, nsteps=10000)

**Vectorization**:

.. code-block:: python

   # Vectorize log_prob_fn
   def vectorized_log_prob(theta_array):
       """
       Evaluate log probability for multiple walkers.

       Parameters
       ----------
       theta_array : array, shape (nwalkers, ndim)

       Returns
       -------
       log_probs : array, shape (nwalkers,)
       """
       return jnp.array([
           bridge.log_prob_fn(theta)
           for theta in theta_array
       ])

Benchmarking Results
--------------------

Typical Performance Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Optimization
     - Speedup
     - Notes
   * - NumPy → JAX
     - 10-50x
     - Model dependent
   * - JAX without JIT → with JIT
     - 2-10x
     - After warmup
   * - CPU → GPU (JAX)
     - 10-1000x
     - Large models, vectorized
   * - Scalar → Vectorized
     - 5-50x
     - Batch evaluation
   * - Python loops → JAX ops
     - 100x+
     - Avoid Python overhead
   * - Matrix inverse → solve
     - 2-5x
     - Numerical stability too
   * - Precomputation
     - 2-100x
     - Problem dependent

Example: End-to-End Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Original (slow): ~100 ms per likelihood
   def slow_model(params):
       A = params['A']
       f = params['f']
       phi = params['phi']

       signal = np.zeros(len(data))
       for i in range(len(data)):
           signal[i] = A * np.sin(2*np.pi*f*times[i] + phi)

       cov = build_covariance(params)
       residual = data - signal
       inv_cov = np.linalg.inv(cov)

       return -0.5 * residual @ inv_cov @ residual

   # Optimized (fast): ~0.1 ms per likelihood
   class FastModel:
       def __init__(self, data, times):
           self.data = jnp.array(data)
           self.times = jnp.array(times)

       def __call__(self, params):
           A = params['A']
           f = params['f']
           phi = params['phi']

           # Vectorized signal
           signal = A * jnp.sin(2*jnp.pi*f*self.times + phi)

           # Precomputed or cached covariance
           cov = self.get_covariance(params)
           residual = self.data - signal

           # Use solve, not inverse
           return -0.5 * residual @ jnp.linalg.solve(cov, residual)

   # JIT-compiled
   model = jit(FastModel(data, times))
   bridge = DiscoveryJAXNSBridge(model, priors, jit=True)
   bridge.configure_array_api(order=['A', 'f', 'phi'])

   # Result: 1000x speedup!

Monitoring Performance
----------------------

Track Sampling Progress
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time

   start = time.time()

   # Nessai: check iteration count
   results = bridge.run_sampler(nlive=1000, output='output/')

   end = time.time()
   runtime = end - start

   # Analyze efficiency
   n_likelihood_calls = results.get('total_likelihood_evaluations', 0)
   time_per_call = runtime / n_likelihood_calls if n_likelihood_calls > 0 else 0

   print(f"Total runtime: {runtime:.1f} s")
   print(f"Likelihood calls: {n_likelihood_calls}")
   print(f"Time per call: {time_per_call*1000:.3f} ms")

Memory Profiling
^^^^^^^^^^^^^^^^

.. code-block:: python

   import tracemalloc

   # Start memory tracking
   tracemalloc.start()

   # Run sampling
   results = bridge.run_sampler(nlive=1000, max_samples=10000)

   # Get peak memory
   current, peak = tracemalloc.get_traced_memory()
   print(f"Peak memory: {peak / 1024**2:.1f} MB")

   tracemalloc.stop()

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
