"""
Discovery ↔︎ GPry bridge (via **Cobaya**)

This version is tailored to **GPry** and Cobaya's model wrapper.
It builds a Cobaya `model` (with your likelihood + priors) and passes that
`model` directly to `gpry.Runner`.

Key references:
- GPry Runner accepts a Cobaya `model` as its first argument; in that case it
  reads priors and names from the model (no `bounds` needed).
- In Cobaya, an external likelihood can be a plain Python function placed under
  the `likelihood` block, and priors are defined in the `params` block using
  scipy.stats names (e.g. `uniform`, `norm`, `loguniform`) or `min`/`max`.

What you get
------------
- `build_cobaya_info(...)` → a ready-to-use Cobaya `info` dict (likelihood + params)
- `get_cobaya_model(info)` → a `cobaya.model.Model`
- `DiscoveryGPryCobayaBridge` → convenience class that:
  1) adapts your Discovery-style likelihood
  2) builds the Cobaya model with proper priors
  3) launches `gpry.Runner(model, ...)` and exposes results

Assumptions
-----------
- Your *Discovery* model is either a callable `loglike(params_dict) -> float`
  or an object with `.log_prob(params_dict)` / `.log_likelihood(params_dict)`.
- You provide priors in a compact mapping per parameter.
  Supported entries here map to Cobaya priors:
    * ("uniform", a, b)  →  `prior: {min: a, max: b}`
    * ("loguniform", a, b)  →  `prior: {dist: loguniform, a: a, b: b}`
    * ("normal", mean, sigma[, ...])  →  `prior: {dist: norm, loc: mean, scale: sigma}`
    * ("fixed", value)  →  `value: value` (non-sampled)
  (These match Cobaya's documented syntax and SciPy parameterization. citeturn6search0)

If you need custom/callable priors, you can add them under Cobaya's top-level
`prior:` block—see the TODO note in `build_cobaya_info`.
"""
from __future__ import annotations

from curses.ascii import alt
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, List

import numpy as np

# --------------------------- Discovery adapter --------------------------- #

class _DiscoveryAdapter:
    def __init__(self, model: Any):
        self.model = model
        if callable(model):
            self._fn = model
        elif hasattr(model, "log_prob") and callable(getattr(model, "log_prob")):
            self._fn = getattr(model, "log_prob")
        elif hasattr(model, "logL") and callable(getattr(model, "logL")):
            self._fn = getattr(model, "logL")
        else:
            raise TypeError(
                "Model must be callable or expose .log_prob/.logL(params_dict)."
            )

    def __call__(self, params: Mapping[str, float]) -> float:
        return float(self._fn(params))


# ------------------------------ Priors ---------------------------------- #

@dataclass
class ParsedPrior:
    kind: str  # uniform | loguniform | normal | fixed
    args: Tuple[float, ...]


def _parse_one(spec: Any) -> ParsedPrior:
    if isinstance(spec, dict):
        tag = str(spec["dist"]).lower()
        if tag == "uniform":
            a, b = spec["min"], spec["max"]
            return ParsedPrior("uniform", (float(a), float(b)))
        if tag == "loguniform":
            a, b = spec["a"], spec["b"]
            return ParsedPrior("loguniform", (float(a), float(b)))
        if tag == "normal":
            mu, sigma = spec["loc"], spec["scale"]
            return ParsedPrior("normal", (float(mu), float(sigma)))
        if tag == "fixed":
            v = spec["value"]
            return ParsedPrior("fixed", (float(v),))
    raise ValueError(f"Unsupported prior spec: {spec}")


def split_priors(priors: Mapping[str, Any]) -> Tuple[List[str], Dict[str, float], Dict[str, ParsedPrior]]:
    sampled: List[str] = []
    fixed: Dict[str, float] = {}
    parsed: Dict[str, ParsedPrior] = {}
    for name, spec in priors.items():
        p = _parse_one(spec)
        parsed[name] = p
        if p.kind == "fixed":
            fixed[name] = p.args[0]
        else:
            sampled.append(name)
    return sampled, fixed, parsed


# --------------------------- Cobaya builders ---------------------------- #

def _make_likelihood_func(parsed_prior: Mapping[str, ParsedPrior], adapter: _DiscoveryAdapter, 
                          alternative_paramnames: Optional[Mapping[str, str]]) -> Callable[..., float]:
    """Return a Cobaya-friendly likelihood.

    Cobaya accepts external likelihood *functions*; it will pass the sampled
    parameters by name as kwargs. We accept **kwargs to be robust to future
    changes in the parameterization. (Cobaya can introspect function args, but
    since we declare params explicitly in the `params` block, **kwargs is fine.)
    See Cobaya's advanced example for function likelihoods. citeturn5view0
    """
    names = list(parsed_prior.keys())
    if alternative_paramnames:
        # Map alternative names back to original names
        orig_to_alt = {k: v for k, v in alternative_paramnames.items()}
        alt_to_orig = {v: k for k, v in alternative_paramnames.items()}
        print(alt_to_orig)
        names = [orig_to_alt.get(n, n) for n in names]
    else:
        alt_to_orig = {n: n for n in names}  # identity mapping
        orig_to_alt = alt_to_orig # identity mapping
    names = tuple(names)
    print("All names:", names)

    def loglike(**kwargs: float) -> float:
        pd: Dict[str, float] = {}
        # Only keep the sampled names we advertised
        for n in names:
            if n not in kwargs:
                raise KeyError(f"Missing parameter '{n}' in likelihood call")
            # Map alternative name back to original name
            n_orig = alt_to_orig[n]
            pd[n_orig] = float(kwargs[n])
        return adapter(pd)
    # Explicitly set the function kwargs for introspection such that Cobaya finds the parameter names
    # loglike.__signature__ = None  # avoid issues if inspect.signature is called
    # loglike.__name__ = "discovery_loglike"
    # loglike.__doc__ = "Log-likelihood function adapted from Discovery model."


    return loglike, names


def _prior_entry_from_parsed(p: ParsedPrior) -> Dict[str, Any] | float:
    if p.kind == "uniform":
        a, b = p.args
        return {"prior": {"min": a, "max": b}}
    if p.kind == "loguniform":
        a, b = p.args
        # Cobaya delegates to SciPy: loguniform takes 'a' (min) and 'b' (max).
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html
        return {"prior": {"dist": "loguniform", "a": a, "b": b}}
    if p.kind == "normal":
        mu, sigma = p.args
        return {"prior": {"dist": "norm", "loc": mu, "scale": sigma}}
    if p.kind == "fixed":
        (v,) = p.args
        return v  # fixed parameter
    raise ValueError(f"Unsupported prior kind: {p.kind}")


def build_cobaya_info(
    *,
    discovery_model: Any,
    priors: Mapping[str, Any],
    latex_labels: Optional[Mapping[str, str]] = None,
    like_name: str = "discovery_like",
    alternative_paramnames: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Create a Cobaya `info` dict with one external likelihood and a `params` block.

    Notes
    -----
    - If you need external/callable *global* priors (beyond 1D priors in params),
      add them to `info['prior']` after this returns (see Cobaya docs). citeturn5view0
    """
    adapter = _DiscoveryAdapter(discovery_model)
    sampled, fixed, parsed = split_priors(priors)

    # Likelihood function (external function style)
    loglike, sampled_names = _make_likelihood_func(parsed, adapter, alternative_paramnames=alternative_paramnames)
    info: Dict[str, Any] = {"likelihood": 
                            {like_name: 
                             {"external": loglike,
                              "input_params": sampled_names}}}

    # Params block
    pblock: Dict[str, Any] = {}
    for name, p in parsed.items():
        entry = _prior_entry_from_parsed(p)
        # add latex label if provided
        if isinstance(entry, dict):

            if latex_labels and name in latex_labels:
                entry = {**entry, "latex": latex_labels[name]}
        if alternative_paramnames and name in alternative_paramnames:
            alternative_name = alternative_paramnames[name]
            name = alternative_name  # for fixed params too
        
        pblock[name] = entry

    info["params"] = pblock

    return info


def get_cobaya_model(info: Mapping[str, Any]):
    from cobaya.model import get_model
    return get_model(info)


# ------------------------------ GPry glue -------------------------------- #

class DiscoveryGPryCobayaBridge:
    """Bridge that:
    - builds a Cobaya model from a Discovery-style likelihood + priors
    - runs GPry by passing the Cobaya model to `gpry.Runner`
    """

    def __init__(
        self,
        discovery_model: Any,
        priors: Mapping[str, Any],
        *,
        latex_labels: Optional[Mapping[str, str]] = None,
        alternative_paramnames: Optional[Mapping[str, str]] = None,
        like_name: str = "discovery_like",
        runner_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.info = build_cobaya_info(
            discovery_model=discovery_model,
            priors=priors,
            latex_labels=latex_labels,
            like_name=like_name,
            alternative_paramnames=alternative_paramnames
        )
        self.all_names = list(self.info["params"].keys())
        self.sampled_names = [n for n, v in self.info["params"].items() if not isinstance(v, (int, float))]
        self.fixed_names = [n for n, v in self.info["params"].items() if isinstance(v, (int, float))]
        self.model = get_cobaya_model(self.info)
        self.runner_kwargs = dict(runner_kwargs or {})
        self.runner = None
        self.results = None

    def run(self, *, checkpoint: Optional[str] = None, **run_kwargs: Any):
        """Create and run `gpry.Runner` with the Cobaya model.

        Per GPry docs, when passing a Cobaya `model` as first argument, GPry
        uses the model's prior and parameter names automatically. citeturn2view0
        """
        from gpry import Runner

        rkwargs = dict(self.runner_kwargs)
        if checkpoint is not None:
            rkwargs.setdefault("checkpoint", checkpoint)
        runner = Runner(self.model, **rkwargs)
        self.runner = runner
        self.results = runner.run(**run_kwargs)
        return self.results

    # Convenience accessors (these use GPry's common post-run attributes)
    def posterior_samples(self) -> Optional[np.ndarray]:
        if self.runner is None:
            return None
        # GPry Cobaya wrapper stores samples via Cobaya products; but the Runner
        # can also expose `surrogate_samples` after MC on the GP. Try common names.
        for attr in ("posterior_samples", "surrogate_samples", "samples"):
            if hasattr(self.runner, attr):
                return np.asarray(getattr(self.runner, attr))
        return None


__all__ = [
    "build_cobaya_info",
    "get_cobaya_model",
    "DiscoveryGPryCobayaBridge",
]
