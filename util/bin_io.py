import struct
import numpy as np


def load_bin_data(file_name: str) -> np.ndarray:
    """Read a .bin file written by save_bin_data.

    Format: groups of 5 complex samples packed as interleaved int16 pairs
    (real then imag) within 32-byte blocks, Q14 scaling.
    """
    with open("./{}.bin".format(file_name), "rb") as f:
        data = f.read()
    n_bytes = len(data)
    decode_data = []
    for i in range(int(n_bytes / 32)):
        for j in range(5):
            s = i * 32 + j * 4
            re = float(struct.unpack("h", data[s : s + 2])[0])
            im = float(struct.unpack("h", data[s + 2 : s + 4])[0])
            decode_data.append((re + 1j * im) / 2**14)
    return np.array(decode_data)
