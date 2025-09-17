import numpy as np

class PowAmpNonLinRapp():
    def __init__(self, gain_db = 38.5, op1db_dbm = 22.1, p_sat_dbm = 34):
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
    
    def power_dbm_to_voltage_peak(self, power_dbm, sys_imp=50):
        power_watt = 10**((power_dbm)/10) * 1e-3
        voltage_rms = np.sqrt(power_watt * sys_imp)
        return np.sqrt(2) * voltage_rms
    
    def voltage_peak_to_power_dbm(self, voltage_peak, sys_imp=50):
        voltage_rms = voltage_peak / np.sqrt(2)
        power_watt = voltage_rms**2 / sys_imp
        return 10*np.log10(power_watt*1e3)