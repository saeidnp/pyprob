import torch
from termcolor import colored

from . import Distribution, Normal
from .. import util
from .truncated_normal_functions import trandn, moments


# Beware: clamp_mean_between_low_high=True prevents derivative computation with respect to mean when it's outside [low, high]
class TruncatedNormal(Distribution):
    def __init__(self, mean_non_truncated, stddev_non_truncated, low, high, clamp_mean_between_low_high=False):
        self._mean_non_truncated = util.to_tensor(mean_non_truncated)
        self._stddev_non_truncated = util.to_tensor(stddev_non_truncated)
        if self._mean_non_truncated.dim() == 0:
            self._mean_non_truncated = self._mean_non_truncated.unsqueeze(0)
        if self._stddev_non_truncated.dim() == 0:
            self._stddev_non_truncated = self._stddev_non_truncated.unsqueeze(0)

        self._low = util.to_tensor(low).view(self._mean_non_truncated.shape)
        self._high = util.to_tensor(high).view(self._mean_non_truncated.shape)
        if clamp_mean_between_low_high:
            self._mean_non_truncated = torch.max(torch.min(self._mean_non_truncated,
                                                           self._high),
                                                 self._low)

        output = moments(self._low, self._high, self._mean_non_truncated,
                         self._stddev_non_truncated**2)
        self._logZ, self._Z, self._mean, self._var, self._entropy = output

        self._standard_normal_dist = Normal(util.to_tensor(torch.zeros_like(self._mean_non_truncated)), util.to_tensor(torch.ones_like(self._stddev_non_truncated)))
        self._alpha = (self._low - self._mean_non_truncated) / self._stddev_non_truncated
        self._beta = (self._high - self._mean_non_truncated) / self._stddev_non_truncated
        batch_shape = self._mean_non_truncated.size()
        event_shape = torch.Size()
        super().__init__(name='TruncatedNormal', address_suffix='TruncatedNormal', batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'TruncatedNormal(mean_non_truncated:{}, stddev_non_truncated:{}, low:{}, high:{})'.format(self._mean_non_truncated, self._stddev_non_truncated, self._low, self._high)

    def log_prob(self, value, sum=False):
        value = util.to_tensor(value)
        tmp = (value - self._mean_non_truncated)/self._stddev_non_truncated
        log_prob = self._standard_normal_dist.log_prob(tmp)\
                   - (torch.log(self._stddev_non_truncated) + self._logZ)
        if util.has_nan_or_inf(log_prob):
            import sys
            mask = torch.isinf(log_prob) | torch.isnan(log_prob)
            print(colored('Warning: NaN, -Inf, or Inf encountered in TruncatedNormal log_prob.', 'red', attrs=['bold']))
            print('distribution', self.name)
            print('value', value[mask])
            print('log_prob', log_prob[mask])
            print('parent_mean', self._mean_non_truncated[mask])
            print('parent_stddev', self._stddev_non_truncated[mask])
            print('log_Z', self._logZ[mask])
            sys.exit()
            # lp = util.replace_inf(lp, colored('Warning: TruncatedNormal log_prob has inf, replacing with 0.', 'red', attrs=['bold']))
        return torch.sum(log_prob) if sum else log_prob

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def mean_non_truncated(self):
        return self._mean_non_truncated

    @property
    def stddev_non_truncated(self):
        return self._stddev_non_truncated

    @property
    def variance_non_truncated(self):
        return self._stddev_non_truncated.pow(2)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._var

    def sample(self):
        sample = trandn(self._alpha, self._beta).view(self._alpha.shape)
        return sample*self._stddev_non_truncated + self._mean_non_truncated

    def to(self, device):
        self._mean_non_truncated.to(device=device)
        self._stddev_non_truncated.to(device=device)
        self._low.to(device=device)
        self._high.to(device=device)
        self.__init__(self._mean_non_truncated, self._stddev_non_truncated,
                      self._low, self._high)
        return self
