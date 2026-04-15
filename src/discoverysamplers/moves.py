"""
Smart proposal moves for Eryn MCMC/RJMCMC sampling.

Provides adaptive and Gibbs-style moves based on Lorenzo Speri's implementations,
adapted for use with the discoverysamplers package.

Key features:
- Adaptive Metropolis (AM) proposals in eigenbasis of the covariance
- Gibbs-style block updates via ``indx_list``
- Periodic prior draws (10% of the time) to help escape local modes
- Periodic parameter wrapping (e.g., angles)
"""

import numpy as np
from eryn.moves.mh import MHMove


class GaussianMove(MHMove):
    """Gaussian proposal with adaptive and block-update features.

    Supports three proposal modes:

    - ``"Gaussian"``: Standard multivariate Gaussian proposal.
    - ``"AM"``: Adaptive Metropolis proposal in the eigenbasis of the
      covariance matrix (SVD-based), with the 2.38/sqrt(d) scaling.

    Additional features enabled via constructor arguments:

    - **Gibbs blocks** (``indx_list``): Each step randomly picks one block
      and only updates those parameter indices.
    - **Prior draws** (``priors``): With 10% probability, all parameters
      for active leaves are redrawn from the prior.
    - **Periodic wrapping**: Handled by Eryn's ``periodic`` kwarg.

    Parameters
    ----------
    cov_all : dict
        Covariance information per branch. Keys are branch names, values can be:
        - scalar: isotropic proposal
        - 1D array: diagonal proposal
        - 2D array: full covariance matrix (used with ``mode``)
    mode : str, optional
        Proposal mode: ``"AM"`` (default) or ``"Gaussian"``.
    factor : float or None, optional
        If given, proposal scale is drawn from
        ``exp(U(-log(factor), log(factor))) * cov``.
    priors : dict or None, optional
        Eryn-format priors ``{branch: {param_index: dist}}``.
        When provided, 10% of proposals redraw all params from the prior.
    indx_list : list of tuples or None, optional
        Gibbs block specification. Each entry is ``(branch_name, bool_mask)``
        where ``bool_mask`` has shape ``(nleaves_max, ndim)`` or ``(ndim,)``
        indicating which parameters to update. ``None`` mask means update all.
    prior_draw_probability : float, optional
        Probability of drawing from the prior instead of the proposal.
        Default 0.1.
    **kwargs
        Passed to ``MHMove`` (e.g., ``periodic``).
    """

    def __init__(
        self,
        cov_all,
        mode="AM",
        factor=None,
        priors=None,
        indx_list=None,
        prior_draw_probability=0.1,
        **kwargs,
    ):
        self.all_proposal = {}

        for name, cov in cov_all.items():
            try:
                float(cov)
            except TypeError:
                cov = np.atleast_1d(cov)
                if len(cov.shape) == 1:
                    ndim = len(cov)
                    proposal = _diagonal_proposal(np.sqrt(cov), factor, mode)
                elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                    ndim = cov.shape[0]
                    if mode == "Gaussian":
                        proposal = _proposal(cov, factor, "vector")
                    elif mode == "AM":
                        proposal = _AM_proposal(cov, factor, "vector")
                    else:
                        raise ValueError(f"Unknown mode '{mode}'. Use 'Gaussian' or 'AM'.")
                else:
                    raise ValueError("Invalid proposal scale dimensions")
            else:
                ndim = None
                proposal = _isotropic_proposal(np.sqrt(cov), factor, mode)

            self.all_proposal[name] = proposal

        self.priors = priors
        self.indx_list = indx_list
        self.prior_draw_probability = prior_draw_probability
        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from Gaussian distribution with smart features.

        Parameters
        ----------
        branches_coords : dict
            Keys are branch names, values are arrays of shape
            ``(ntemps, nwalkers, nleaves_max, ndim)``.
        random : numpy.random.RandomState
            Random state for reproducibility.
        branches_inds : dict or None
            Keys are branch names, values are boolean arrays of shape
            ``(ntemps, nwalkers, nleaves_max)`` indicating active leaves.
        **kwargs
            Ignored (for compatibility).

        Returns
        -------
        tuple
            ``(proposed_coords, factors)`` where factors is zeros (symmetric proposal).
        """
        q = {}
        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            proposal_fn = self.all_proposal[name]
            inds_here = np.where(inds)

            q[name] = coords.copy()

            # Get proposed points from the proposal distribution
            new_coords_tmp = proposal_fn(coords[inds_here], random)[0]
            new_coords = coords[inds_here].copy()

            # Apply Gibbs block updates if specified
            if self.indx_list is not None:
                indx_list_here = [el[1] for el in self.indx_list if el[0] == name]
                nw = new_coords_tmp.shape[0]
                for i in range(nw):
                    temp_ind = random.randint(len(indx_list_here))
                    if indx_list_here[temp_ind] is not None:
                        new_coords[i, indx_list_here[temp_ind][0]] = (
                            new_coords_tmp[i, indx_list_here[temp_ind][0]]
                        )
                    else:
                        new_coords[i, :] = new_coords_tmp[i, :]
            else:
                new_coords = new_coords_tmp.copy()

            # Prior draw with given probability
            if self.priors is not None and name in self.priors:
                if random.uniform() > (1.0 - self.prior_draw_probability):
                    for var in range(new_coords.shape[-1]):
                        if var in self.priors[name]:
                            new_coords[:, var] = self.priors[name][var].rvs(
                                size=new_coords[:, var].shape[0]
                            )

            q[name][inds_here] = new_coords.copy()

        # Handle periodic parameters
        if self.periodic is not None:
            for name, tmp in q.items():
                ntemps, nwalkers, nleaves_max, ndim = tmp.shape
                wrapped = self.periodic.wrap(
                    {name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)}
                )
                q[name] = wrapped[name].reshape(ntemps, nwalkers, nleaves_max, ndim)

        return q, np.zeros((ntemps, nwalkers))


# ---------------------------------------------------------------------------
# Internal proposal helpers
# ---------------------------------------------------------------------------

class _isotropic_proposal:
    allowed_modes = ["vector", "random", "sequential"]

    def __init__(self, scale, factor, mode):
        self.index = 0
        self.scale = scale
        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(
                f"'{mode}' is not a recognized mode. Please select from: {self.allowed_modes}"
            )
        self.mode = mode

    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))

    def __call__(self, x0, rng):
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0)
        if self.mode == "random":
            m = (range(nw), rng.randint(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)


class _diagonal_proposal(_isotropic_proposal):
    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))


class _proposal(_isotropic_proposal):
    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * rng.multivariate_normal(
            np.zeros(len(self.scale)), self.scale, size=len(x0)
        )


class _AM_proposal(_isotropic_proposal):
    """Adaptive Metropolis proposal in the SVD eigenbasis."""

    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        return _propose_AM(x0, rng, self.scale, self.get_factor(rng))


def _propose_AM(x0, rng, cov, scale):
    """Adaptive jump proposal in the eigenbasis of the covariance.

    Performs SVD on the covariance, jumps in the uncorrelated basis
    with the standard 2.38/sqrt(d) scaling, then rotates back.
    """
    new_pos = x0.copy()
    nw, nd = new_pos.shape
    U, S, _ = np.linalg.svd(cov)

    # Transform to eigenbasis
    y = np.dot(U.T, x0.T).T

    # Jump in all eigendirections with proper scaling
    ind_vec = np.arange(nd)
    rng.shuffle(ind_vec)
    y[:, ind_vec] += (
        scale * rng.normal(size=nw)[:, None] * np.sqrt(S[None, ind_vec]) * 2.38 / np.sqrt(nd)
    )

    # Transform back
    new_pos = np.dot(U, y.T).T
    return new_pos


__all__ = ["GaussianMove"]
