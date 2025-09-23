import numpy as np
import json
from pow_amp_non_lin_common import PowAmpNonLinCommon

class PowAmpNonLinLut(PowAmpNonLinCommon):
    def __init__(self):
        with open("lut_amam_ampm.json", "r", encoding="UTF-8") as f:
            lut = json.load(f)
        self.input_volt_peak = self.power_dbm_to_voltage_peak(np.array(lut["p_in"]))
        self.output_volt_peak = self.power_dbm_to_voltage_peak(np.array(lut["am_am"]))
        self.output_phase_rad = (np.array(lut["am_pm"]) - lut["am_pm"][0]) / 180 * np.pi
    
    def apply(self, input):
        index = self.find_index(abs(input))
        return self.output_volt_peak[index] * np.exp(1j*(np.angle(input) + self.output_phase_rad[index]))
    
    def find_index(self, input):
        return np.argmin(np.abs(self.input_volt_peak - input))

