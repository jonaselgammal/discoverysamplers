# Filename: nessai.py
# Author: Jonas El Gammal
# Description: Interface to the nessai nested sampling package
# Date Created: 2025-09-01

import numpy as np

try:
    import nessai
    import nessai.model
    from nessai.flowsampler import FlowSampler

except ImportError:
    raise ImportError("nessai is not installed. Please install it to use this module.")


class nessai_model(nessai.model.Model):
    """
    Generic Model class for the sampling algorithm

    Parameters
    ----------
    log_like_func: func
        should be albertos.sampler_ln_likelihood

    transforms: ParamTransform object
        All the parameter transformations needed to go from the sampling space
        to the waveform evaluation space

    prior_functions: dict
        dictionary from the config file which specifies priors (+kwargs)
        applied to specific parameters of specific source sets
        eg: {'WD1': {'Amplitude': {'function': truncated_powerlaw,
                                   'index': 4}}}
        where the 'Amplitude' param of source set 'WD1' is taking the
        truncated_powerlaw prior, which has kwarg 'index=4'

    kwargs: dict
        Keyword arguments for log_likelihood


    Returns
    -------

    cls: class
        This Model class, containing data and kwargs

    livepoint: class
        contains all parameters, likelihoods, etc required for the sampling

    """

    def __init__(self, log_like_func, log_pri_func, transforms, prior_functions, **kwargs):

        self.log_like_func = log_like_func       # actual likelihood function
        self.log_pri_func = log_pri_func       # actual likelihood function
        self.transforms = transforms             # ParamTransforms object
        self.names = transforms.sampled_params   # names of all params
        self.like_kwargs = kwargs    # kwargs for likelihood function
        self.log_prior_functions = prior_functions    # list of non-flat priors
        self.prior_indices = []      # names of params to apply the priors to
        self.pri_kwargs = kwargs    # kwargs for likelihood function
        
        self.bounds = {}
        for i in range(len(self.names)):
            self.bounds[self.names[i]] = transforms.sampled_bounds[i]


        

    def log_likelihood(self, livepoint):
        """
        Wrapper for log-likelihood function. Performs the appropriate
        transformations into the physical parameters, then calls the albertos
        log-likelihood function
        """
        # array giving all the parameter values in order
        # FIXME: should add a new transform that takes a dict as input
        ll = np.zeros(livepoint.size)
        
        if livepoint.ndim == 0:
            point = np.array([livepoint[p] for p in self.names])
            source_values = self.transforms.sampled_to_waveform(point)
            ll = self.log_like_func(source_values, **self.like_kwargs)
        else:
            for i in range(livepoint.size):
                point = np.array([livepoint[i][p] for p in self.names])
                source_values = self.transforms.sampled_to_waveform(point)
                ll[i] = self.log_like_func(source_values, **self.like_kwargs)
                
        return ll


    def log_prior(self, livepoint):
        """
        Generic log-prior function. Evaluates the priors on the params
        """
        if not self.in_bounds(livepoint).any():
            # Inherited method from nessai to discard points outside bounds
            return -np.inf

        # Evaluate the joint prior
        lp = np.log(self.in_bounds(livepoint))

        if self.log_prior_functions is None:
            pass
        else:
            if livepoint.ndim == 0:
                point = np.array([livepoint[p] for p in self.names])
                source_values = self.transforms.sampled_to_waveform(point)
                lp += self.log_pri_func(source_values, self.transforms.subset_names, self.log_prior_functions)
            else:
                for i in range(livepoint.size):
                    point = np.array([livepoint[i][p] for p in self.names])
                    source_values = self.transforms.sampled_to_waveform(point)
                    lp[i] += self.log_pri_func(source_values, self.transforms.subset_names, self.log_prior_functions)

        return lp