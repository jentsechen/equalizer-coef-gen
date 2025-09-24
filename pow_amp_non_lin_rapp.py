import numpy as np
from pow_amp_non_lin_common import PowAmpNonLinCommon

class PowAmpNonLinRapp(PowAmpNonLinCommon):
    def __init__(self, gain_db = 35, op1db_dbm = 22.1, p_sat_dbm = 34):
        self.gain_db = gain_db
        self.op1db_dbm = op1db_dbm
        self.p_sat_dbm = p_sat_dbm
        self.gain = self.set_gain(gain_db)
        self.ov1db = self.power_dbm_to_voltage_peak(op1db_dbm)
        self.v_sat = self.power_dbm_to_voltage_peak(p_sat_dbm)
        self.p = 0.699

    def set_gain(self, gain_db):
        return 10**(gain_db/20)
    
    def apply(self, input):
        return self.amam(abs(input)) * np.exp(1j*np.angle(input))

    def amam(self, in_mag):
        return (self.gain * in_mag) / (1 + (self.gain*in_mag/self.v_sat)**(2*self.p))**(1/(2*self.p))
    