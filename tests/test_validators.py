import pytest
import typing
import datetime
import dataclasses
import uuid
import enum
import inspect
from collections import OrderedDict
from meow.validators import *
from .mypy_test_helper import *


def test_string():
    assert V[str] is V[str]
    assert V[str] == String()
    assert V[str].validate("123") == "123"
    assert reveal_type(V[str]) == "meow.validators.elements.Validator[builtins.str*]"
    assert reveal_type(V[str].validate("123")) == "builtins.str*"
    with pytest.raises(ValidationError):
        V[str].validate(123)
    with pytest.raises(ValidationError):
        V[str].validate(None)
    validator = V(str, min_length=2, max_length=10, pattern="a+")
    assert reveal_type(validator) == "meow.validators.elements.Validator[builtins.str*]"
    assert validator == String(min_length=2, max_length=10, pattern="a+")
    assert validator.validate("aaa") == "aaa"
    with pytest.raises(ValidationError):
        validator.validate("a")
    with pytest.raises(ValidationError):
        validator.validate("aaaaaaaaaaa")
    with pytest.raises(ValidationError):
        validator.validate("bbbbb")
    with pytest.raises(ValidationError):
        String(min_length=1).validate("")


def test_int():
    assert V[int] is V[int]
    assert V[int] == Integer()
    assert V[int].validate(123) == 123
    assert V[int].validate("123", allow_coerce=True) == 123
    assert reveal_type(V[int]) == "meow.validators.elements.Validator[builtins.int*]"
    assert reveal_type(V[int].validate(123)) == "builtins.int*"
    with pytest.raises(ValidationError):
        V[int].validate("123")
    with pytest.raises(ValidationError):
        V[int].validate(123.2)
    with pytest.raises(ValidationError):
        V[int].validate(True)
    with pytest.raises(ValidationError):
        V[int].validate(None)
    validator = V(int, minimum=2, maximum=10)
    assert validator == Integer(minimum=2, maximum=10)
    assert validator.validate(2) == 2
    assert validator.validate(10) == 10
    assert reveal_type(validator) == "meow.validators.elements.Validator[builtins.int*]"
    with pytest.raises(ValidationError):
        validator.validate(1)
    with pytest.raises(ValidationError):
        validator.validate(11)
    with pytest.raises(ValidationError):
        validator.validate("asd", allow_coerce=True)
    try:
        validator.validate("asd", allow_coerce=True)
    except ValidationError as e:
        assert e.as_dict() == {"": "Must be a number."}
    validator = V(
        int, minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True
    )
    assert validator == Integer(
        minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True
    )
    assert validator.validate(3) == 3
    assert validator.validate(9) == 9
    with pytest.raises(ValidationError):
        validator.validate(2)
    with pytest.raises(ValidationError):
        validator.validate(10)


def test_float():
    assert V[float] is V[float] and V[float] == Float()
    assert V[float].validate(123.4) == 123.4
    assert V[float].validate("123.4", allow_coerce=True) == 123.4
    revealed_type = reveal_type(V[float])
    assert revealed_type == "meow.validators.elements.Validator[builtins.float*]"
    revealed_type = reveal_type(V[float].validate(123))
    assert revealed_type == "builtins.float*"
    with pytest.raises(ValidationError):
        V[float].validate("123.4")
    with pytest.raises(ValidationError):
        V[float].validate(True)
    with pytest.raises(ValidationError):
        V[float].validate(None)
    validator = V(float, minimum=2, maximum=10)
    assert validator == Float(minimum=2, maximum=10)
    assert validator.validate(2) == 2
    assert validator.validate(10) == 10
    with pytest.raises(ValidationError):
        validator.validate(1.999)
    with pytest.raises(ValidationError):
        validator.validate(10.001)
    validator = V(
        float, minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True
    )
    assert validator == Float(
        minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True
    )
    assert validator.validate(2.00001) == 2.00001
    assert validator.validate(9.99999) == 9.99999
    with pytest.raises(ValidationError):
        validator.validate(2)
    with pytest.raises(ValidationError):
        validator.validate(10)


def test_bool():
    assert V[bool] is V[bool] and V[bool] == Boolean()
    assert V[bool].validate(True) is True
    assert V[bool].validate(False) is False
    assert V[bool].validate("on", allow_coerce=True) is True
    assert V[bool].validate("off", allow_coerce=True) is False
    assert V[bool].validate("1", allow_coerce=True) is True
    assert V[bool].validate("0", allow_coerce=True) is False
    assert V[bool].validate("true", allow_coerce=True) is True
    assert V[bool].validate("false", allow_coerce=True) is False

    revealed_type = reveal_type(V[bool])
    assert revealed_type == "meow.validators.elements.Validator[builtins.bool*]"
    revealed_type = reveal_type(V[bool].validate("1", allow_coerce=True))
    assert revealed_type == "builtins.bool*"

    with pytest.raises(ValidationError):
        V[bool].validate(None)
    with pytest.raises(ValidationError):
        V[bool].validate(1)
    with pytest.raises(ValidationError):
        V[bool].validate("xxx", allow_coerce=True)


def test_datetime():
    assert (
        V[datetime.datetime] is V[datetime.datetime]
        and V[datetime.datetime] == DateTime()
    )
    assert V[datetime.datetime].validate("2020-01-02T01:02:03") == datetime.datetime(
        2020, 1, 2, 1, 2, 3
    )
    assert V[datetime.datetime].validate("2020-01-02T01:02:03Z") == datetime.datetime(
        2020, 1, 2, 1, 2, 3, tzinfo=datetime.timezone.utc
    )
    assert V[datetime.datetime].validate(
        "2020-01-02T01:02:03+01:00"
    ) == datetime.datetime(
        2020, 1, 2, 1, 2, 3, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600))
    )
    assert V[datetime.datetime].validate(
        "2020-01-02T01:02:03-01:00"
    ) == datetime.datetime(
        2020, 1, 2, 1, 2, 3, tzinfo=datetime.timezone(datetime.timedelta(seconds=-3600))
    )

    revealed_type = reveal_type(V[datetime.datetime])
    assert revealed_type == "meow.validators.elements.Validator[datetime.datetime*]"
    revealed_type = reveal_type(
        V[datetime.datetime].validate("2020-01-02T01:02:03-01:00")
    )
    assert revealed_type == "datetime.datetime*"

    with pytest.raises(ValidationError):
        V[datetime.datetime].validate(None)
    with pytest.raises(ValidationError):
        V[datetime.datetime].validate(1)
    with pytest.raises(ValidationError):
        V[datetime.datetime].validate("2020-01-02")


def test_date():
    assert V[datetime.date] is V[datetime.date] and V[datetime.date] == Date()
    assert V[datetime.date].validate("2020-01-02") == datetime.date(2020, 1, 2)

    revealed_type = reveal_type(V[datetime.date])
    assert revealed_type == "meow.validators.elements.Validator[datetime.date*]"
    revealed_type = reveal_type(V[datetime.date].validate("2020-01-02"))
    assert revealed_type == "datetime.date*"

    with pytest.raises(ValidationError):
        V[datetime.date].validate(None)
    with pytest.raises(ValidationError):
        V[datetime.date].validate(1)
    with pytest.raises(ValidationError):
        V[datetime.date].validate("2020-01-02T01:01:01")


def test_time():
    assert V[datetime.time] is V[datetime.time] and V[datetime.time] == Time()
    assert V[datetime.time].validate("01:02:03") == datetime.time(1, 2, 3)

    revealed_type = reveal_type(V[datetime.time])
    assert revealed_type == "meow.validators.elements.Validator[datetime.time*]"
    revealed_type = reveal_type(V[datetime.time].validate("01:02:03"))
    assert revealed_type == "datetime.time*"

    with pytest.raises(ValidationError):
        V[datetime.time].validate(None)
    with pytest.raises(ValidationError):
        V[datetime.time].validate(1)
    with pytest.raises(ValidationError):
        V[datetime.time].validate("25:01:01")


def test_uuid():
    assert V[uuid.UUID] is V[uuid.UUID] and V[uuid.UUID] == UUID()
    assert V[uuid.UUID].validate("8dd8ceb7-3d5c-4f46-b932-e7ba4078f7cf") == uuid.UUID(
        "8dd8ceb7-3d5c-4f46-b932-e7ba4078f7cf"
    )

    revealed_type = reveal_type(V[uuid.UUID])
    assert revealed_type == "meow.validators.elements.Validator[uuid.UUID*]"
    revealed_type = reveal_type(
        V[uuid.UUID].validate("8dd8ceb7-3d5c-4f46-b932-e7ba4078f7cf")
    )
    assert revealed_type == "uuid.UUID*"

    with pytest.raises(ValidationError):
        V[uuid.UUID].validate(None)
    with pytest.raises(ValidationError):
        V[uuid.UUID].validate(1)
    with pytest.raises(ValidationError):
        V[uuid.UUID].validate("xdd8ceb7-3d5c-4f46-b932-e7ba4078f7cf")


def test_optional():
    # mypy==0.780: union of types is not type anymore (
    assert V[typing.Optional[str]] is V[typing.Optional[str]]
    assert V[typing.Optional[str]] == Optional(String())
    assert V[typing.Optional[str]].validate(None) is None
    assert V[typing.Optional[str]].validate("123") == "123"


def test_union():
    assert V[typing.Union[str, int]] is V[typing.Union[str, int]]
    assert V[typing.Union[str, int]] == Union((String(), Integer()))
    assert V[typing.Union[str, int]].validate("asd") == "asd"
    assert V[typing.Union[str, int]].validate(123) == 123
    with pytest.raises(ValidationError):
        V[typing.Union[str, int]].validate(123.3)


def test_any():
    assert V[typing.Any] is V[typing.Any] and V[typing.Any] == Any
    assert V[typing.Any].validate("asd") == "asd"
    assert V[typing.Any].validate(1) == 1
    assert V[typing.Any].validate(True) == True
    assert V[typing.Any].validate(None) is None


def test_mapping():
    assert V[dict] is V[dict] and V[dict] == Mapping(Any, Any)
    assert V[dict].validate({"a": 10}) == {"a": 10}

    revealed_type = reveal_type(V[dict])
    assert (
        revealed_type == "meow.validators.elements.Validator[builtins.dict*[Any, Any]]"
    )
    revealed_type = reveal_type(V[dict].validate({"a": 10}))
    assert revealed_type == "builtins.dict*[Any, Any]"

    with pytest.raises(ValidationError):
        V[dict].validate("asd")

    validator = Mapping(Any, Any, min_items=2, max_items=3)
    with pytest.raises(ValidationError):
        validator.validate({"a": 10})
    with pytest.raises(ValidationError):
        validator.validate({"a": 10, "b": 10, "c": 10, "d": 10})

    validator = V[typing.Mapping[str, int]]
    revealed_type = reveal_type(validator)
    assert (
        revealed_type
        == "meow.validators.elements.Validator[def () -> typing.Mapping[builtins.str*, builtins.int*]]"
    )
    revealed_type = reveal_type(validator.validate({"a": 10}))
    assert revealed_type == "def () -> typing.Mapping[builtins.str*, builtins.int*]"

    with pytest.raises(ValidationError):
        validator.validate({"a": "b"})
    with pytest.raises(ValidationError):
        validator.validate({1: 10})
    assert validator.validate({"a": 1}) == {"a": 1}

    assert isinstance(
        TypedMapping(String(), String(), converter=OrderedDict).validate(
            {"a": "b"}
        ),
        OrderedDict,
    )


def test_object():
    validator = Object({"a": Integer(), "b": String()})
    assert (
        reveal_type(validator.validate({"a": 10, "b": "asd"}))
        == "typing.Mapping[builtins.str, Any]"
    )


def test_list():
    assert V[list] is V[list] and V[list] == List(Any)
    assert V[list].validate([1, "a", True]) == [1, "a", True]
    with pytest.raises(ValidationError):
        V[list].validate("asd")

    revealed_type = reveal_type(V[list])
    assert revealed_type == "meow.validators.elements.Validator[builtins.list*[Any]]"
    revealed_type = reveal_type(V[list].validate([]))
    assert revealed_type == "builtins.list*[Any]"

    validator = List(Any, min_items=2, max_items=3)

    with pytest.raises(ValidationError):
        validator.validate([1])
    with pytest.raises(ValidationError):
        validator.validate([1, 2, 3, 4])

    validator = V[typing.List[str]]
    with pytest.raises(ValidationError):
        validator.validate(["a", 1])
    assert validator.validate(["a", "b"]) == ["a", "b"]
    revealed_type = reveal_type(validator)
    assert (
        revealed_type
        == "meow.validators.elements.Validator[builtins.list*[builtins.str*]]"
    )

    validator = List(Any, unique_items=True)
    with pytest.raises(ValidationError):
        validator.validate(["a", {"a": 1}, {"a": 1}, [1], [1], True, False])

    assert validator.validate(["a", {"a": 1}, [1], True, False]) == [
        "a",
        {"a": 1},
        [1],
        True,
        False,
    ]


def test_tuple():
    assert V[tuple] == Tuple(Any)
    assert (
        reveal_type(V[tuple])
        == "meow.validators.elements.Validator[builtins.tuple*[Any]]"
    )
    assert V[typing.Tuple] == Tuple(Any)
    assert (
        reveal_type(V[tuple])
        == "meow.validators.elements.Validator[builtins.tuple*[Any]]"
    )
    assert V[typing.Tuple[int, ...]] == Tuple(Integer())
    assert reveal_type(V[typing.Tuple[int, ...]])  # broken in mypy 0.780

    assert V[typing.Tuple[int]] == TypedTuple((Integer(),))
    assert V[typing.Tuple[int, int]] == TypedTuple((Integer(), Integer()))
    assert V[typing.Tuple[int, str]] == TypedTuple((Integer(), String()))

    assert V[typing.Tuple[int, str]].validate([1, "s"]) == (1, "s")

    with pytest.raises(ValidationError):
        V[typing.Tuple[int, str]].validate("asdads")

    with pytest.raises(ValidationError):
        V[typing.Tuple[int, str]].validate([1, 1])


def test_sets():
    assert V[set] == Set(Any)
    assert V[frozenset] == FrozenSet(Any)
    assert V[typing.AbstractSet] == FrozenSet(Any)
    assert V[typing.Set] == Set(Any)
    assert V[typing.MutableSet] == Set(Any)
    assert V[typing.FrozenSet] == FrozenSet(Any)
    assert V[typing.Set[str]] == Set(V[str])
    assert V[typing.MutableSet[str]] == Set(V[str])
    assert V[typing.FrozenSet[str]] == FrozenSet(V[str])

    assert (
        reveal_type(V[set]) == "meow.validators.elements.Validator[builtins.set*[Any]]"
    )
    assert (
        reveal_type(V[frozenset])
        == "meow.validators.elements.Validator[builtins.frozenset*[Any]]"
    )
    assert (
        reveal_type(V[typing.Set[str]])
        == "meow.validators.elements.Validator[builtins.set*[builtins.str*]]"
    )
    assert (
        reveal_type(V[typing.MutableSet[str]])
        == "meow.validators.elements.Validator[def () -> typing.MutableSet[builtins.str*]]"
    )
    assert (
        reveal_type(V[typing.FrozenSet[str]])
        == "meow.validators.elements.Validator[builtins.frozenset*[builtins.str*]]"
    )
    assert V[typing.Set[str]].validate(["a", "b"]) == {"a", "b"}
    with pytest.raises(ValidationError):
        V[typing.Set[str]].validate(["a", "a"])


def test_sequences():
    assert V[typing.Sequence[str]] == Tuple(V[str])
    assert (
        reveal_type(V[typing.Sequence[str]])
        == "meow.validators.elements.Validator[def () -> typing.Sequence[builtins.str*]]"
    )


def test_other():
    with pytest.raises(TypeError):
        x = V[typing.Generator]


class Enum1(enum.Enum):
    A = 1
    B = 2


def test_enum():
    assert V[Enum1] is V[Enum1]
    assert V[Enum1].validate("A") == Enum1.A

    assert (
        reveal_type(V[Enum1])
        == "meow.validators.elements.Validator[tests.test_validators.Enum1*]"
    )
    assert reveal_type(V[Enum1].validate("A")) == "tests.test_validators.Enum1*"

    with pytest.raises(ValidationError):
        V[Enum1].validate("C")
    with pytest.raises(ValidationError):
        V[Enum1].validate(1)


@dataclasses.dataclass
class A:
    x: str
    y: str = "a"
    z: str = field(init=False, default="aaa")


@dataclasses.dataclass
class B:
    i: int
    j: float


@dataclasses.dataclass
class C:
    a: typing.Tuple[A, B]
    b: str = field(min_length=2)
    d: typing.Tuple[str, str, int] = field(
        items=(String(), String(min_length=1), Integer(maximum=3))
    )
    e: typing.Union[int, str] = field(items=(Integer(), String(min_length=1)))
    c: typing.Optional[typing.Mapping[str, str]] = field(min_items=2, default=None)


def test_dataclasses():
    assert V[C] is V[C]
    assert V[C] == TypedObject(
        {
            "a": TypedTuple(
                (
                    TypedObject(
                        {"x": String(), "y": String()}, converter=A, required=("x",)
                    ),
                    TypedObject(
                        {"i": Integer(), "j": Float()},
                        converter=B,
                        required=("i", "j"),
                    ),
                )
            ),
            "b": String(min_length=2),
            "c": Optional(Mapping(String(), String(), min_items=2)),
            "d": TypedTuple((String(), String(min_length=1), Integer(maximum=3))),
            "e": Union((Integer(), String(min_length=1))),
        },
        converter=C,
        required=("a", "b", "d", "e"),
    )
    assert V[C].validate(
        {
            "a": [{"x": "aaa"}, {"i": 1, "j": 1.1}],
            "b": "ccc",
            "d": ["xxx", "x", 2],
            "e": "22",
        }
    ) == C(a=(A(x="aaa"), B(i=1, j=1.1)), b="ccc", d=("xxx", "x", 2), e="22")
    assert (
        reveal_type(V[C])
        == "meow.validators.elements.Validator[tests.test_validators.C*]"
    )

    with pytest.raises(ValidationError):
        V[C].validate({})
    with pytest.raises(ValidationError):
        V[C].validate([])
    with pytest.raises(ValidationError):
        V[C].validate({1: 1})
    with pytest.raises(ValidationError):
        V[C].validate({"a": [], "b": "ccc"})

    try:
        V[C].validate({"a": [], "b": "ccc", "e": 1})
    except ValidationError as e:
        assert e.as_dict() == {
            "a": "Must have exact 2 items.",
            "d": 'The "d" field is required.',
        }


class SpecialType(int):
    pass


T = typing.TypeVar("T")


class MyGenericType(typing.Generic[T]):
    def __init__(self, value: typing.List[T]):
        self.value = value
        # and all other methods

    def get_some(self, key: int) -> T:
        return self.value[key]


def resolve_default(obj, v_args):
    if obj is SpecialType:
        return lambda **spec: Integer(minimum=1, maximum=2, **spec)
    origin = typing.get_origin(obj)
    if origin is MyGenericType and len(v_args) == 1:
        return lambda **spec: TypedList(v_args[0], converter=MyGenericType, **spec)


def test_default():
    V = Container(default=resolve_default)
    assert V[SpecialType] is V[SpecialType]
    assert V[SpecialType].validate(1) == 1
    assert (
        reveal_type(V[SpecialType])
        == "meow.validators.elements.Validator[tests.test_validators.SpecialType*]"
    )
    assert (
        reveal_type(V[SpecialType].validate(1)) == "tests.test_validators.SpecialType*"
    )

    assert V[MyGenericType[int]]

    with pytest.raises(ValidationError):
        V[SpecialType].validate(0)

    with pytest.raises(TypeError):
        assert V[typing.Generic[int, int]]
