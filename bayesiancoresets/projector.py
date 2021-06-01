import numpy as np
from .util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts, pts):
        raise NotImplementedError

class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(np.array([]), np.array([]))

    def project(self, pts, grad=False):
        lls = self.loglikelihood(pts, self.samples)
        lls -= lls.mean(axis=1)[:,np.newaxis]
        if grad:
            if self.grad_loglikelihood is None:
                raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            glls = self.grad_loglikelihood(pts, self.samples)
            glls -= glls.mean(axis=2)[:, :, np.newaxis]
            return lls, glls
        else:
            return lls

    def update(self, wts, pts):
        self.samples = self.sampler(self.projection_dimension, wts, pts)
        
## for projection, we don't actually need the samples, just the log likelihood
## evaluated at the samples
class StanFitProjector(Projector):
    def __init__(self, sampler, N, projection_dimension):
        """
        Inputs:
            sampler: lambda function to return the log-likelihoods
            N: scalar, original number of observations
            projection_dimension: scalar, number of samples to draw to discretize
                log-likelihoods
        """
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.N = N
        self.update(np.array([]), np.array([]))
        
    def project(self, pts):
        """
        Depending on length of pts, return projection on either the full
        data set or the coreset data.
        """
        if (len(pts) == self.N):
            lls = self.full_lls.copy()
        else:
            lls = self.wts_lls.copy()
        lls -= lls.mean(axis=1)[:,np.newaxis]
        return lls

    def update(self, wts, pts):
        self.full_lls, self.wts_lls = self.sampler(self.projection_dimension, wts, pts)