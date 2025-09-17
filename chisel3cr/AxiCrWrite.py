from typing import List, Protocol, Tuple, Union
from util.ToJsonIO import ToJson, Validator, Convertor
from bitstring import BitArray, BitStream
from threading import Lock


class AxiCrWrite(Protocol):
    write_lock = Lock()

    @staticmethod
    def __pad2words__(data: BitArray, wordwidth: int = 32) -> BitArray:
        n = (data.length % wordwidth)
        x = data.length + (32 - n) if n != 0 else 0
        return data + BitArray('0b0' * (x - data.length))

    def write(self, addr: int, data: BitArray) -> None:
        with self.write_lock:
            self.__write__(addr, self.__pad2words__(data))

    def __write__(self, addr: int, data: BitArray) -> None:
        raise NotImplementedError("please implement __write__ method")


class DummyAxiCrWrite(AxiCrWrite):

    def __write__(self, addr: int, data: BitArray) -> None:
        print(f'write: addr={addr}, data={data.hex}')
