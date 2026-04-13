import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from chisel3cr.common import qt, eSign, eMSB, eLSB

def main():
    with open("aaf_freq_resp_256.json", "r", encoding="UTF-8") as f:
        aaf_freq_resp_j = json.load(f)
    aaf_freq_resp_list = []
    for i in range(9):
        aaf_freq_resp_list.append(aaf_freq_resp_j["re"][i] + 1j*np.array(aaf_freq_resp_j["im"]))

if __name__ == "__main__":
    main()