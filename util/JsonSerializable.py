import json
from dataclasses import dataclass, asdict, field, is_dataclass, fields
from typing import Type, TypeVar, Any, Dict, List, Union, get_origin, get_args, Optional
from datetime import datetime
from enum import Enum, auto
import numpy as np

T = TypeVar("T", bound="JsonIO")

FLOAT_TYPE_SUPPORT_LIST = (float, int)
COMPLEX_TYPE_SUPPORT_LIST = complex


def non_optional_type(t: type) -> type:
    if get_origin(t) == Union and type(None) in get_args(t):
        for arg in get_args(t):
            if arg is not type(None):
                return arg
    return t


def str2cpx(s: str) -> complex:
    re, im = s.replace("(", "").replace(")", "").split(",")
    return complex(float(re), float(im))


class JsonIO:
    def __post_init__(self):
        self.__validate_fields()
        self.do_post_init()

    def validate_fields(self): ...

    def do_post_init(self): ...

    def __validate_fields(self):
        for field_info in fields(self):  # type: ignore
            value = getattr(self, field_info.name)
            self._validate_type(field_info.type, value, field_info.name)

        self.validate_fields()

    def _validate_type(self, expected_type: Any, value: Any, field_name: str):
        origin = get_origin(expected_type)
        if origin:
            if origin == list:
                if not isinstance(value, list):
                    raise TypeError(f"Field '{field_name}' is expected to be a list.")
                inner_type = get_args(expected_type)[0]
                for item in value:
                    self._validate_type(inner_type, item, f"{field_name}[]")
            elif origin == dict:
                if not isinstance(value, dict):
                    raise TypeError(f"Field '{field_name}' is expected to be a dict.")
                key_type, value_type = get_args(expected_type)
                for k, v in value.items():
                    self._validate_type(key_type, k, f"{field_name}{{key}}")
                    self._validate_type(value_type, v, f"{field_name}{{value}}")
            elif origin == Union and type(None) in get_args(expected_type):
                if value is not None:
                    self._validate_type(non_optional_type(expected_type), value, field_name)
            else:
                raise TypeError(f"Unsupported origin type: {origin} for field '{field_name}'.")
        else:
            if expected_type == datetime:
                if not isinstance(value, datetime):
                    raise TypeError(f"Field '{field_name}' is expected to be a datetime.")
            elif isinstance(expected_type, type) and issubclass(expected_type, Enum):
                if not type(value).__name__ == expected_type.__name__:
                    raise TypeError(f"Field '{field_name}' is expected to be an instance of {expected_type.__name__}.")
            elif expected_type == complex:
                if not isinstance(value, COMPLEX_TYPE_SUPPORT_LIST):
                    raise TypeError(f"Field '{field_name}' is expected to be an instance of {expected_type.__name__}")
            elif expected_type == np.ndarray:
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"Field '{field_name}' is expected to be an ndarray.")
            elif expected_type is float:
                if not isinstance(value, FLOAT_TYPE_SUPPORT_LIST):
                    raise TypeError(f"Field '{field_name}' is expected to be an instance of {expected_type.__name__} or int.")
            elif expected_type is int:
                if not isinstance(value, int):
                    raise TypeError(f"Field '{field_name}' is expected to be an instance of {expected_type.__name__} or int.")
            else:
                if not isinstance(value, expected_type):
                    Warning(f"Field '{field_name}' is expected to be an instance of {expected_type.__name__}.")
                    # raise TypeError(f"Field '{field_name}' is expected to be an instance of {expected_type.__name__}.")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=self._default, indent=4)

    def to_dict(self) -> Dict[str, Any]:
        result = {}

        for field_info in fields(self):  # type: ignore
            value = getattr(self, field_info.name)
            if value is not None:
                result[field_info.name] = self.__formatting__(value, non_optional_type(field_info.type))

        return result

    def __formatting__(self, value: Any, field_type: type):

        if isinstance(value, complex):
            return f"({value.real}, {value.imag})"
        elif isinstance(value, np.ndarray):
            return self.__formatting_ndarray(value)
        elif isinstance(value, Enum):
            return value.name
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(field_type, type) and issubclass(field_type, JsonIO):
            return value.to_dict()
        elif get_origin(field_type) == list:
            return self.__formatting_list(value, non_optional_type(get_args(field_type)[0]))
        elif isinstance(field_type, dict):
            return self.__formatting_dict(value, field_type)
        # elif is_dataclass(value):
        #     return asdict(value)
        else:
            return value

    @staticmethod
    def __formatting_ndarray(value: np.ndarray):
        def __format__(x):
            if isinstance(x, complex):
                return f"({x.real}, {x.imag})"
            if isinstance(x, np.complexfloating):
                return f"({x.real}, {x.imag})"
            if isinstance(x, np.ndarray):
                return JsonIO.__formatting_ndarray(x)
            return x

        return [__format__(item) for item in value]

    def __formatting_list(self, value: List[Any], field_type: type):
        return [self.__formatting__(item, field_type) for item in value]

    def __formatting_dict(self, value_dict: Dict[Any, Any], field_type: type):
        key_type, value_type = get_args(field_type)
        value_type = non_optional_type(value_type)
        return {key: self.__formatting__(value, value_type) for key, value in value_dict.items() if value is not None}

    def save_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, default=self._default, indent=4)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        data = json.loads(json_str)
        return cls._from_dict(data)

    @classmethod
    def load_json(cls: Type[T], file_path: str) -> T:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        field_types = {f.name: non_optional_type(f.type) for f in cls.__dataclass_fields__.values()}  # type: ignore
        data_out = {}

        for field_name, field_type in field_types.items():

            if field_name not in data:
                continue

            if isinstance(data[field_name], str):
                if field_type == datetime:
                    # data[field_name] = datetime.fromisoformat(data[field_name])
                    data_out[field_name] = datetime.fromisoformat(data[field_name])
                elif field_type == complex:
                    # data[field_name] = str2cpx(data[field_name])
                    data_out[field_name] = str2cpx(data[field_name])
                elif isinstance(field_type, type) and issubclass(field_type, Enum):
                    # data[field_name] = field_type[data[field_name]]
                    data_out[field_name] = field_type[data[field_name]]

            elif isinstance(data[field_name], list):
                if isinstance(field_type, type) and issubclass(field_type, np.ndarray):
                    # data[field_name] = cls._convert_ndarray(data[field_name], field_type)
                    data_out[field_name] = cls._convert_ndarray(data[field_name], field_type)
                else:
                    # data[field_name] = cls._convert_list(data[field_name], field_type)
                    data_out[field_name] = cls._convert_list(data[field_name], field_type)
            elif isinstance(data[field_name], dict):
                if isinstance(field_type, type) and issubclass(field_type, JsonIO):
                    # data[field_name] = field_type._from_dict(data[field_name])
                    data_out[field_name] = field_type._from_dict(data[field_name])
                else:
                    # data[field_name] = cls._convert_dict(data[field_name], field_type)
                    data_out[field_name] = cls._convert_dict(data[field_name], field_type)
            else:
                # data[field_name] = field_type(data[field_name])
                data_out[field_name] = field_type(data[field_name])

        return cls(**data_out)

    @staticmethod
    def _convert_ndarray(data_list: List[Any], field_type: Any) -> np.ndarray:
        def _convert_(x):
            if isinstance(x, list):
                return np.array([_convert_(item) for item in x])
            if "(" in x and ")" in x:
                return str2cpx(x)
            return x

        return np.array([_convert_(item) for item in data_list])

    @staticmethod
    def _convert_list(data_list: List[Any], field_type: Any) -> List[Any]:
        origin = get_origin(field_type)
        if origin == list:
            inner_type = non_optional_type(get_args(field_type)[0])
            if isinstance(inner_type, type) and issubclass(inner_type, JsonIO):
                return [inner_type._from_dict(item) for item in data_list]
            elif inner_type == complex:
                return [str2cpx(value) for value in data_list]
            elif isinstance(inner_type, type) and issubclass(inner_type, Enum):
                return [inner_type[item] for item in data_list]
            elif isinstance(inner_type, type) and issubclass(inner_type, np.ndarray):
                return [np.array(item) for item in data_list]
            elif get_origin(inner_type) == list:
                return [JsonIO._convert_list(item, inner_type) for item in data_list]
            else:
                return [inner_type(item) for item in data_list]
        return data_list

    @staticmethod
    def _convert_dict(data_dict: Dict[Any, Any], field_type: Any) -> Dict[Any, Any]:
        origin = get_origin(field_type)
        if origin == dict:
            key_type, value_type = get_args(field_type)
            value_type = non_optional_type(value_type)
            new_dict = {}
            for key, value in data_dict.items():
                new_key = key_type(key)
                if isinstance(value, dict) and isinstance(value_type, type) and issubclass(value_type, JsonIO):
                    new_value = value_type._from_dict(value)
                elif isinstance(value_type, type) and issubclass(value_type, Enum):
                    new_value = value_type[value]  # type: ignore
                elif isinstance(value_type, type) and issubclass(value_type, np.ndarray):
                    new_value = np.array(value)
                elif isinstance(value, list) and get_origin(value_type) == list:
                    new_value = JsonIO._convert_list(value, value_type)
                else:
                    new_value = value_type(value)  # type: ignore
                new_dict[new_key] = new_value
            return new_dict
        return data_dict

    @staticmethod
    def _default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, Enum):
            return o.name
        elif isinstance(o, complex):
            return (o.real, o.imag)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, JsonIO):
            return o.to_dict()
        elif isinstance(o, list):
            return [JsonIO._default(item) for item in o]
        elif is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


if __name__ == "__main__":

    # 定義 Enum 類別
    class Color(Enum):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    # 範例內嵌的 JsonSerializable 類別
    @dataclass
    class Address(JsonIO):
        city: Optional[str] = None
        zipcode: Optional[str] = None

    # 範例使用
    @dataclass
    class ExampleDataClass(JsonIO):
        name: Optional[str] = None
        age: Optional[int] = None
        birthdate: Optional[datetime] = None
        favorite_color: Optional[Color] = None
        complex_list: Optional[List[complex]] = None
        address: Optional[Address] = None
        nested_json_list: Optional[List[Address]] = None
        color_list: Optional[List[Color]] = None
        list_of_lists: Optional[List[List[int]]] = None
        address_map: Optional[Dict[str, Address]] = None
        color_map: Optional[Dict[str, Color]] = None
        ndarray_field: Optional[np.ndarray] = None

    example = ExampleDataClass(
        name="Alice",
        age=30,
        favorite_color=Color.GREEN,
        complex_list=[3 + 5j, 4 + 3j],
        address=Address(city="Wonderland", zipcode="12345"),
        nested_json_list=[Address(city="City1", zipcode="11111"), Address(city="City2", zipcode="22222")],
        color_list=[Color.RED, Color.BLUE],
        list_of_lists=[[1, 2], [3, 4]],
        address_map={"home": Address(city="HomeCity", zipcode="00001"), "work": Address(city="WorkCity", zipcode="00002")},
        color_map={"primary": Color.RED, "secondary": Color.BLUE},
        ndarray_field=np.array([[[complex(1), complex(2), complex(3)], [4, 5, 6]]]),
    )

    example.save_json("example.json")

    example_loaded_from_file = ExampleDataClass.load_json("example.json")
    print(example_loaded_from_file)
