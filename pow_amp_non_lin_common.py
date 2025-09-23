import numpy as np

class PowAmpNonLinCommon():
    def power_dbm_to_voltage_peak(self, power_dbm, sys_imp=50):
        power_watt = 10**((power_dbm)/10) * 1e-3
        voltage_rms = np.sqrt(power_watt * sys_imp)
        return np.sqrt(2) * voltage_rms
    
    def voltage_peak_to_power_dbm(self, voltage_peak, sys_imp=50):
        voltage_rms = voltage_peak / np.sqrt(2)
        power_watt = voltage_rms**2 / sys_imp
        return 10*np.log10(power_watt*1e3)