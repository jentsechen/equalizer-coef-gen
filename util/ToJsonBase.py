from enum import Enum
import numpy as np
import json
from typing import Any


class ToJsonBase:

    def __post_init__(self):
        pass

    def from_dict(self, d: dict):
        self.__init__(**self.convert_x(d))
        return self

    def to_dict(self) -> dict:
        raise NotImplementedError("virtualMethod is virutal! Must be overwrited.")
        return self.__dict__()

    def to_dict_seriesable(self):
        return self.formatting(self.to_dict())

    def to_json(self) -> str:
        return json.dumps(self.to_dict_seriesable())

    def save_json(self, filename: str):
        with open(filename, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def formatting(cls, x):
        if isinstance(x, complex):
            return "({}, {})".format(x.real, x.imag)
        if isinstance(x, np.int8):
            return int(x)
        if isinstance(x, np.int16):
            return int(x)
        if isinstance(x, np.int32):
            return int(x)
        if isinstance(x, np.int64):
            return int(x)

        if isinstance(x, dict):
            return {k: cls.formatting(v) for k, v in x.items() if v is not None}
        if isinstance(x, list):
            return [cls.formatting(v) for v in x if v is not None]
        if isinstance(x, Enum):
            return x.name
        if isinstance(x, np.bool_ or bool):
            return 1 if x else 0
        if isinstance(x, np.ndarray):
            return cls.formatting(x.tolist())
        if isinstance(x, ToJsonBase):
            return x.to_dict_seriesable()

        return x

    @classmethod
    def convert_x(cls, d: Any) -> Any:
        if isinstance(d, str):
            # if d.count("j") == 1:
            #     return complex(d.replace(" ", ""))
            if d.startswith("(") and d.endswith(")"):
                s1 = d.replace('(', '').replace(')', '').split(',')
                return complex(float(s1[0]), float(s1[1]))
        if isinstance(d, list):
            return [cls.convert_x(x) for x in d if x is not None]
        if isinstance(d, dict):
            return {k: cls.convert_x(v) for k, v in d.items() if v is not None}
        return d

    @classmethod
    def make_serializale(cls, x):
        return cls.formatting(x)

    def load_json(self, filename: str):
        with open(filename, 'r') as f:
            data = f.read()
        return self.from_json(data)

    def from_json(self, data: str):
        return self.from_dict(json.loads(data))
