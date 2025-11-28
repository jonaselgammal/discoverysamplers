# Quick Start Guide (script form)
#
# Mirrors the notebook example, but runnable as a plain Python script so we
# can test quickly without needing Jupyter in this sandbox.

import json
import re
import numpy as np
import jax
import discovery as ds
from discoverysamplers.priors import standard_priors

# Tidy up threading / cache to avoid sandbox issues
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

mpl_dir = os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

# Toggle samplers for this script run
RUN_NESSAI = False
RUN_JAXNS = False
RUN_GPRY = False
RUN_ERYN = False

jax.config.update("jax_enable_x64", True)

# Path to a bundled pulsar file (relative to the discovery package install)
path_to_discovery = ds.__path__[0]
PULSAR_FEATHER = f"{path_to_discovery}/../../data/v1p1_de440_pint_bipm2019-J2322+2057.feather"
print(f"Using pulsar file: {PULSAR_FEATHER}")

# Load pulsar and build a likelihood
psr = ds.Pulsar.read_feather(PULSAR_FEATHER)
likelihood = ds.PulsarLikelihood([
    psr.residuals,
    ds.makenoise_measurement(psr),
])
params = likelihood.logL.params
print(f"Discovered {len(params)} parameters")
print("Parameters:", params)

# Standard priors from discovery (uniform bounds defined in priordict_standard)
priors = standard_priors(params)

# Confirm every parameter matched Discovery's priordict_standard
import discovery.prior as dp
matched = {}
for name in params:
    pattern = next((pat for pat in dp.priordict_standard if re.match(pat, name)), None)
    if pattern is None:
        raise KeyError(f"No standard prior found for parameter '{name}'.")
    bounds = dp.priordict_standard[pattern]
    spec = priors[name]
    if (spec["dist"], spec["min"], spec["max"]) != ("uniform", float(bounds[0]), float(bounds[1])):
        raise ValueError(f"Mismatch between priordict_standard and parsed prior for '{name}'.")
    matched[name] = pattern

print(f"Matched {len(matched)} parameters to Discovery priordict_standard.")
print("Sample of priors:")
print(json.dumps({k: priors[k] for k in params[:3]}, indent=2))

# Quick sanity check: draw one sample from the priors and evaluate the likelihood
rng = np.random.default_rng(123)
sample = {}
for name, spec in priors.items():
    dist = spec["dist"]
    if dist == "uniform":
        sample[name] = rng.uniform(spec["min"], spec["max"])
    elif dist == "loguniform":
        sample[name] = np.exp(rng.uniform(np.log(spec["min"]), np.log(spec["max"])))
    elif dist == "fixed":
        sample[name] = spec["value"]
    else:
        raise ValueError(f"Unsupported dist in this quick demo: {dist}")
print("Log-likelihood for one random draw:", likelihood.logL(sample))

# Nessai (optional)
if RUN_NESSAI:
    try:
        from discoverysamplers.nessai_interface import DiscoveryNessaiBridge

        bridge = DiscoveryNessaiBridge(discovery_model=likelihood.logL, priors=priors, jit=True)
        results = bridge.run_sampler(
            nlive=100,
            max_iteration=300,
            output="examples/nessai_quick_out",
            resume=False,
        )
        print("Nessai logZ:", results["logZ"], "+/-", results.get("logZ_err"))
    except Exception as exc:
        print("Nessai run skipped due to error:", exc)
else:
    print("RUN_NESSAI=False; skipping Nessai demo")

# JAX-NS (optional)
if RUN_JAXNS:
    try:
        from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge

        bridge = DiscoveryJAXNSBridge(discovery_model=likelihood.logL, priors=priors, latex_labels=None, jit=True)
        results = bridge.run_sampler(nlive=200, max_samples=2000, termination_frac=0.05, rng_seed=0)
        print("JAX-NS logZ:", results["logZ"], "+/-", results.get("logZerr"))
    except Exception as exc:
        print("JAX-NS run skipped due to error:", exc)
else:
    print("RUN_JAXNS=False; skipping JAX-NS demo")

# Eryn (optional)
if RUN_ERYN:
    try:
        from discoverysamplers.eryn_interface import DiscoveryErynBridge

        eryn_priors = {}
        for name, spec in priors.items():
            if spec["dist"] == "uniform":
                eryn_priors[name] = {"dist": "uniform", "min": spec["min"], "max": spec["max"]}
            elif spec["dist"] == "loguniform":
                eryn_priors[name] = {"dist": "loguniform", "a": spec["min"], "b": spec["max"]}
            elif spec["dist"] == "fixed":
                eryn_priors[name] = {"dist": "fixed", "value": spec["value"]}
            else:
                raise ValueError("Unsupported prior for Eryn demo")

        bridge = DiscoveryErynBridge(model=likelihood, priors=eryn_priors)
        nwalkers = max(2 * bridge.ndim, 32)
        sampler = bridge.create_sampler(nwalkers=nwalkers)
        p0 = bridge.eryn_prior_container.rvs(size=nwalkers)
        sampler.run_mcmc(p0, nsteps=10, progress=False)
        print("Eryn chain shape:", sampler.get_chain().shape)
    except Exception as exc:
        print("Eryn run skipped due to error:", exc)
else:
    print("RUN_ERYN=False; skipping Eryn demo")

# GPry (optional)
if RUN_GPRY:
    try:
        from discoverysamplers.gpry_interface import DiscoveryGPryCobayaBridge

        bridge = DiscoveryGPryCobayaBridge(discovery_model=likelihood.logL, priors=priors, like_name="pulsar_likelihood")
        info, sampler = bridge.run_sampler(max_samples=2000)
        products = sampler.products()
        print("GPry outputs:", list(products.keys()))
    except Exception as exc:
        print("GPry run skipped due to error:", exc)
else:
    print("RUN_GPRY=False; skipping GPry demo")
