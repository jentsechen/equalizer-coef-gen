import numpy as np
from pow_amp_non_lin_common import PowAmpNonLinCommon

class PowAmpNonLinSaleh(PowAmpNonLinCommon):
    def __init__(self, alpha_a=2.1587, beta_a=1.1517, alpha_phi=4.0033, beta_phi=9.1040):
        self.alpha_a = alpha_a
        self.beta_a = beta_a
        self.alpha_phi = alpha_phi
        self.beta_phi = beta_phi

    def amam(self, in_mag):
        return (self.alpha_a * in_mag) / (1 + self.beta_a * in_mag**2)
    
    def ampm(self, in_mag):
        return (self.alpha_phi * in_mag**2) / (1 + self.beta_phi * in_mag**2)

    def apply(self, input):
        in_mag = abs(input)
        return self.amam(in_mag) * np.exp(1j*(np.angle(input)+self.ampm(in_mag)))