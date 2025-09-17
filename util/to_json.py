import json

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from enum import Enum
import numpy as np
# dataclass = dataclass
# dataclass_json = dataclass_json
from ToJsonBase import ToJsonBase
from JsonIO import ToJson as ToJson2


@dataclass
class ToJson(ToJsonBase):

    def from_dict(self, data: dict):
        return self

    def to_json(self):
        return ""

    def to_json_formatting(self):
        return json.dumps(self.to_dict_seriesable())

    def save_json(self, filename: str):
        with open(filename, 'w') as f:
            f.write(self.to_json_formatting())

    def parse_json(self, data: str):
        return self.parse_dict(json.loads(data))

    def parse_dict(self, data: dict):
        return self.make_serializale(self.from_dict(data))

    # def load_json(self, filename: str):
    #     with open(filename, 'r') as f:
    #         return self.parse_json(f.read())

    @staticmethod
    def to_cpx(d):
        if isinstance(d, str):
            s1 = d.replace('(', '').replace(')', '').split(',')
            return complex(float(s1[0]), float(s1[1]))
        if isinstance(d, list):
            return np.array([ToJson.to_cpx(x) for x in d])
        return d


if __name__ == '__main__':

    @dataclass_json
    @dataclass
    class A(ToJson):
        x: int = 5
        y: float = 3.0

    A(x=3, y=100.2).save_json('test.json')
    b = A.load_json('test.json')
    c = A.from_json(b.to_json())
    print(b, c)
