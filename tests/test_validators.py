import pytest  # type: ignore
import typing
import datetime
import dataclasses
import uuid
import enum
from collections import OrderedDict, deque
from meow.validators import *


def test_string():
    assert V[str] is V(str) and V[str] == String()
    assert V[str].validate("123") == "123"
    with pytest.raises(ValidationError):
        V[str].validate(123)
    with pytest.raises(ValidationError):
        V[str].validate(None)
    v = V(str, min_length=2, max_length=10, pattern="a+")
    assert v == String(min_length=2, max_length=10, pattern="a+")
    assert v.validate("aaa") == "aaa"
    with pytest.raises(ValidationError):
        v.validate("a")
    with pytest.raises(ValidationError):
        v.validate("aaaaaaaaaaa")
    with pytest.raises(ValidationError):
        v.validate("bbbbb")
    with pytest.raises(ValidationError):
        String(min_length=1).validate("")


def test_int():
    assert V[int] is V(int) and V[int] == Integer()
    assert V[int].validate(123) == 123
    assert V[int].validate("123", allow_coerce=True) == 123
    with pytest.raises(ValidationError):
        V[int].validate("123")
    with pytest.raises(ValidationError):
        V[int].validate(123.2)
    with pytest.raises(ValidationError):
        V[int].validate(True)
    with pytest.raises(ValidationError):
        V[int].validate(None)
    v = V(int, minimum=2, maximum=10)
    assert v == Integer(minimum=2, maximum=10)
    assert v.validate(2) == 2
    assert v.validate(10) == 10
    with pytest.raises(ValidationError):
        v.validate(1)
    with pytest.raises(ValidationError):
        v.validate(11)
    with pytest.raises(ValidationError):
        v.validate("asd", allow_coerce=True)
    try:
        v.validate("asd", allow_coerce=True)
    except ValidationError as e:
        assert e.as_dict() == {"": "Must be a number."}
    v = V(int, minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True)
    assert v == Integer(
        minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True
    )
    assert v.validate(3) == 3
    assert v.validate(9) == 9
    with pytest.raises(ValidationError):
        v.validate(2)
    with pytest.raises(ValidationError):
        v.validate(10)


def test_float():
    assert V[float] is V(float) and V[float] == Float()
    assert V[float].validate(123.4) == 123.4
    assert V[float].validate("123.4", allow_coerce=True) == 123.4
    with pytest.raises(ValidationError):
        V[float].validate("123.4")
    with pytest.raises(ValidationError):
        V[float].validate(True)
    with pytest.raises(ValidationError):
        V[float].validate(None)
    v = V(float, minimum=2, maximum=10)
    assert v == Float(minimum=2, maximum=10)
    assert v.validate(2) == 2
    assert v.validate(10) == 10
    with pytest.raises(ValidationError):
        v.validate(1.999)
    with pytest.raises(ValidationError):
        v.validate(10.001)
    v = V(float, minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True)
    assert v == Float(
        minimum=2, maximum=10, exclusive_minimum=True, exclusive_maximum=True
    )
    assert v.validate(2.00001) == 2.00001
    assert v.validate(9.99999) == 9.99999
    with pytest.raises(ValidationError):
        v.validate(2)
    with pytest.raises(ValidationError):
        v.validate(10)


def test_bool():
    assert V[bool] is V(bool) and V[bool] == Boolean()
    assert V[bool].validate(True) is True
    assert V[bool].validate(False) is False
    assert V[bool].validate("on", allow_coerce=True) is True
    assert V[bool].validate("off", allow_coerce=True) is False
    assert V[bool].validate("1", allow_coerce=True) is True
    assert V[bool].validate("0", allow_coerce=True) is False
    assert V[bool].validate("true", allow_coerce=True) is True
    assert V[bool].validate("false", allow_coerce=True) is False

    with pytest.raises(ValidationError):
        V[bool].validate(None)
    with pytest.raises(ValidationError):
        V[bool].validate(1)
    with pytest.raises(ValidationError):
        V[bool].validate("xxx", allow_coerce=True)


def test_datetime():
    assert (
        V[datetime.datetime] is V(datetime.datetime)
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
    with pytest.raises(ValidationError):
        V[datetime.datetime].validate(None)
    with pytest.raises(ValidationError):
        V[datetime.datetime].validate(1)
    with pytest.raises(ValidationError):
        V[datetime.datetime].validate("2020-01-02")


def test_date():
    assert V[datetime.date] is V(datetime.date) and V[datetime.date] == Date()
    assert V[datetime.date].validate("2020-01-02") == datetime.date(2020, 1, 2)
    with pytest.raises(ValidationError):
        V[datetime.date].validate(None)
    with pytest.raises(ValidationError):
        V[datetime.date].validate(1)
    with pytest.raises(ValidationError):
        V[datetime.date].validate("2020-01-02T01:01:01")


def test_time():
    assert V[datetime.time] is V(datetime.time) and V[datetime.time] == Time()
    assert V[datetime.time].validate("01:02:03") == datetime.time(1, 2, 3)
    with pytest.raises(ValidationError):
        V[datetime.time].validate(None)
    with pytest.raises(ValidationError):
        V[datetime.time].validate(1)
    with pytest.raises(ValidationError):
        V[datetime.time].validate("25:01:01")


def test_uuid():
    assert V[uuid.UUID] is V(uuid.UUID) and V[uuid.UUID] == UUID()
    assert V[uuid.UUID].validate("8dd8ceb7-3d5c-4f46-b932-e7ba4078f7cf") == uuid.UUID(
        "8dd8ceb7-3d5c-4f46-b932-e7ba4078f7cf"
    )
    with pytest.raises(ValidationError):
        V[uuid.UUID].validate(None)
    with pytest.raises(ValidationError):
        V[uuid.UUID].validate(1)
    with pytest.raises(ValidationError):
        V[uuid.UUID].validate("xdd8ceb7-3d5c-4f46-b932-e7ba4078f7cf")


def test_optional():
    assert V[typing.Optional[str]] is V(typing.Optional[str]) and V[
        typing.Optional[str]
    ] == Optional(String())
    assert V[typing.Optional[str]].validate(None) is None
    assert V[typing.Optional[str]].validate("123") == "123"


def test_union():
    assert V[typing.Union[str, int]] is V(typing.Union[str, int]) and V[
        typing.Union[str, int]
    ] == Union([String(), Integer()])
    assert V[typing.Union[str, int]].validate("asd") == "asd"
    assert V[typing.Union[str, int]].validate(123) == 123
    with pytest.raises(ValidationError):
        V[typing.Union[str, int]].validate(123.3)


def test_any():
    assert V[typing.Any] is V(typing.Any) and V[typing.Any] == Any()
    assert V[typing.Any].validate("asd") == "asd"
    assert V[typing.Any].validate(1) == 1
    assert V[typing.Any].validate(True) == True
    assert V[typing.Any].validate(None) is None


def test_mapping():
    assert V[dict] is V(dict) and V[dict] == Mapping()
    assert V[dict].validate({"a": 10}) == {"a": 10}
    assert V[OrderedDict] is V(OrderedDict) and V[OrderedDict] == Mapping(
        cast=OrderedDict
    )
    assert V[OrderedDict].validate({"a": 10}) == OrderedDict({"a": 10})
    with pytest.raises(ValidationError):
        V[dict].validate("asd")
    v = Mapping(min_items=2, max_items=3)
    with pytest.raises(ValidationError):
        v.validate({"a": 10})
    with pytest.raises(ValidationError):
        v.validate({"a": 10, "b": 10, "c": 10, "d": 10})
    v = Mapping(keys=V[str], values=V[int])
    with pytest.raises(ValidationError):
        v.validate({"a": "b"})
    with pytest.raises(ValidationError):
        v.validate({1: 10})
    assert v.validate({"a": 1}) == {"a": 1}


def test_array():
    assert V[list] is V(list) and V[list] == Array()
    assert V[list].validate([1, "a", True]) == [1, "a", True]
    with pytest.raises(ValidationError):
        V[list].validate("asd")
    v = Array(min_items=2, max_items=3)
    with pytest.raises(ValidationError):
        v.validate([1])
    with pytest.raises(ValidationError):
        v.validate([1, 2, 3, 4])
    v = Array(items=[V[str], V[int], V[bool]])
    with pytest.raises(ValidationError):
        v.validate([1])
    with pytest.raises(ValidationError):
        v.validate(["a", 1, True, False])
    assert v.validate(["a", 1, True]) == ["a", 1, True]
    v = Array(items=V[str])
    with pytest.raises(ValidationError):
        v.validate(["a", 1])
    assert v.validate(["a", "b"]) == ["a", "b"]
    v = Array(unique_items=True)
    with pytest.raises(ValidationError):
        v.validate(["a", {"a": 1}, {"a": 1}, [1], [1], True, False])
    assert v.validate(["a", {"a": 1}, [1], True, False]) == [
        "a",
        {"a": 1},
        [1],
        True,
        False,
    ]


def test_tuple():
    assert V[tuple] == Array(cast=tuple)
    assert V[typing.Tuple] == Array(cast=tuple)
    assert V[typing.Tuple[int]] == Array(items=[Integer()], cast=tuple)
    assert V[typing.Tuple[int, int]] == Array(items=[Integer(), Integer()], cast=tuple)
    assert V[typing.Tuple[int, int]] == Array(items=[Integer(), Integer()], cast=tuple)
    assert V[typing.Tuple[int, ...]] == Array(items=Integer(), cast=tuple)


def test_sets():
    assert V[set] == Array(unique_items=True, cast=set)
    assert V[frozenset] == Array(unique_items=True, cast=frozenset)
    assert V[typing.Set[str]] == Array(items=V[str], unique_items=True, cast=set)
    assert V[typing.MutableSet[str]] == Array(items=V[str], unique_items=True, cast=set)
    assert V[typing.FrozenSet[str]] == Array(
        items=V[str], unique_items=True, cast=frozenset
    )


def test_sequences():
    assert V[typing.Sequence[str]] == Array(items=V[str])
    assert V[typing.Deque[str]] == Array(items=V[str], cast=deque)


def test_other():
    with pytest.raises(TypeError):
        V[typing.Generator]


def test_enum():
    class Enum1(enum.Enum):
        A = 1
        B = 2

    assert V[Enum1] is V[Enum1]
    assert V[Enum1].validate("A") == Enum1.A
    with pytest.raises(ValidationError):
        V[Enum1].validate("C")
    with pytest.raises(ValidationError):
        V[Enum1].validate(1)


def test_dataclasses():
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
        c: typing.Optional[typing.Mapping[str, str]] = field(min_items=2, default=None)

    assert V[C] is V(C)
    assert V[C] == Object(
        properties={
            "a": Array(
                items=[
                    Object(
                        properties={"x": String(), "y": String()},
                        required=["x"],
                        cast=A,
                    ),
                    Object(
                        properties={"i": Integer(), "j": Float()},
                        required=["i", "j"],
                        cast=B,
                    ),
                ],
                cast=tuple,
            ),
            "b": String(min_length=2),
            "c": Optional(Mapping(keys=String(), values=String(), min_items=2)),
        },
        required=["a", "b"],
        cast=C,
    )
    assert V[C].validate({"a": [{"x": "aaa"}, {"i": 1, "j": 1.1}], "b": "ccc"}) == C(
        a=(A(x="aaa"), B(i=1, j=1.1)), b="ccc"
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
        V[C].validate({"a": [], "b": "ccc"})
    except ValidationError as e:
        assert e.as_dict() == {"a": "Must have at least 2 items."}


def test_default():
    class SpecialType(int):
        pass

    def special_type(tp):
        if tp is SpecialType:
            return Integer, {"minimum": 1, "maximum": 3}
        raise TypeError()

    V = Container(default=special_type)
    assert V[SpecialType] is V[SpecialType]
    assert V[SpecialType].validate(1) == 1
    with pytest.raises(ValidationError):
        V[SpecialType].validate(0)
