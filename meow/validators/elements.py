# meow.validators
#
# Copyright (c) 2020-present Andrey Churin (aachurin@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import re
import datetime
import uuid
import typing
from .exception import ValidationError


_T = typing.TypeVar("_T")
_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")
_T_co = typing.TypeVar("_T_co", covariant=True)


class _ValidatorMixin(typing.Protocol):
    def error(self, code: str, **context: object) -> typing.NoReturn:
        ...  # pragma: nocover

    def error_message(self, code: str, **context: object) -> str:
        ...  # pragma: nocover


class Validator(typing.Generic[_T]):

    errors: typing.Dict[str, str] = {}

    def error(self, code: str, **context: object) -> typing.NoReturn:
        raise ValidationError(self.error_message(code, **context))

    def error_message(self, code: str, **context: object) -> str:
        return self.errors[code].format_map(context)

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__


class Optional(Validator[typing.Optional[_T]]):
    def __init__(self, validator: Validator[_T]):
        assert isinstance(validator, Validator)
        self.validator = validator

    def validate(
        self, value: object, allow_coerce: bool = False
    ) -> typing.Optional[_T]:
        if value is None:
            return None
        return self.validator.validate(value, allow_coerce)


class String(Validator[str]):
    errors = {
        "type": "Must be a string.",
        "blank": "Must not be blank.",
        "max_length": "Must have no more than {max_length} characters.",
        "min_length": "Must have at least {min_length} characters.",
        "pattern": "Must match the pattern /{pattern}/.",
    }

    def __init__(
        self,
        min_length: typing.Optional[object] = None,
        max_length: typing.Optional[object] = None,
        pattern: typing.Optional[object] = None,
    ):

        assert max_length is None or isinstance(max_length, int)
        assert min_length is None or isinstance(min_length, int)
        assert pattern is None or isinstance(pattern, str)

        self.max_length: typing.Optional[int] = max_length
        self.min_length: typing.Optional[int] = min_length
        self.pattern: typing.Optional[str] = pattern

    def validate(self, value: object, allow_coerce: bool = False) -> str:
        if not isinstance(value, str):
            self.error("type")

        if self.min_length is not None and len(value) < self.min_length:
            if self.min_length == 1:
                self.error("blank")
            else:
                self.error("min_length", min_length=self.min_length)

        if self.max_length is not None and len(value) > self.max_length:
            self.error("max_length", max_length=self.max_length)

        if self.pattern is not None and not re.search(self.pattern, value):
            self.error("pattern", pattern=self.pattern)

        return value


class NumericType(Validator[_T]):
    errors = {
        "type": "Must be a number.",
        "integer": "Must be an integer.",
        "minimum": "Must be greater than or equal to {value}.",
        "exclusive_minimum": "Must be greater than {value}.",
        "maximum": "Must be less than or equal to {value}.",
        "exclusive_maximum": "Must be less than {value}.",
    }

    numeric_type: typing.ClassVar[typing.Type[_T]]

    def __init__(
        self,
        minimum: typing.Optional[object] = None,
        maximum: typing.Optional[object] = None,
        exclusive_minimum: object = False,
        exclusive_maximum: object = False,
    ):

        assert minimum is None or isinstance(minimum, (int, float))
        assert maximum is None or isinstance(maximum, (int, float))
        assert isinstance(exclusive_minimum, bool)
        assert isinstance(exclusive_maximum, bool)

        self.minimum: typing.Optional[float] = minimum
        self.maximum: typing.Optional[float] = maximum
        self.exclusive_minimum: bool = exclusive_minimum
        self.exclusive_maximum: bool = exclusive_maximum

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        if (
            self.numeric_type is int
            and isinstance(value, float)
            and not value.is_integer()
        ):
            self.error("integer")
        elif not allow_coerce and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value is None
        ):
            self.error("type")

        try:
            value = self.numeric_type(value)  # type: ignore
        except (TypeError, ValueError):
            self.error("type")

        if self.minimum is not None:
            if self.exclusive_minimum:
                if value <= self.minimum:
                    self.error("exclusive_minimum", value=self.minimum)
            else:
                if value < self.minimum:
                    self.error("minimum", value=self.minimum)

        if self.maximum is not None:
            if self.exclusive_maximum:
                if value >= self.maximum:
                    self.error("exclusive_maximum", value=self.maximum)
            else:
                if value > self.maximum:
                    self.error("maximum", value=self.maximum)

        return value


class Float(NumericType[float]):
    numeric_type = float


class Integer(NumericType[int]):
    numeric_type = int


class Boolean(Validator[bool]):
    errors = {"type": "Must be a valid boolean."}

    values = {
        "on": True,
        "off": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    def validate(self, value: object, allow_coerce: bool = False) -> bool:
        if not isinstance(value, bool):
            if allow_coerce and isinstance(value, str):
                try:
                    value = self.values[value.lower()]
                except KeyError:
                    self.error("type")
            else:
                self.error("type")
        return value


class DateTimeType(Validator[_T]):
    errors = {"type": "Must be a valid datetime."}

    datetime_pattern: typing.ClassVar[typing.Pattern[str]]
    datetime_type: typing.ClassVar[typing.Type[_T]]

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        if not isinstance(value, str):
            self.error("type")

        match = self.datetime_pattern.match(value)
        if not match:
            self.error("type")

        group = match.groupdict()
        if "microsecond" in group:
            group["microsecond"] = group["microsecond"] and group["microsecond"].ljust(
                6, "0"
            )

        tz = group.pop("tzinfo", None)

        if tz == "Z":
            tzinfo: typing.Optional[datetime.tzinfo] = datetime.timezone.utc

        elif tz is not None:
            offset_minutes = int(tz[-2:]) if len(tz) > 3 else 0
            offset_hours = int(tz[1:3])
            delta = datetime.timedelta(hours=offset_hours, minutes=offset_minutes)
            if tz[0] == "-":
                delta = -delta
            tzinfo = datetime.timezone(delta)

        else:
            tzinfo = None

        kwargs: typing.Dict[str, object] = {
            k: int(v) for k, v in group.items() if v is not None
        }
        if tzinfo is not None:
            kwargs["tzinfo"] = tzinfo

        try:
            value = self.datetime_type(**kwargs)  # type: ignore
        except ValueError:
            self.error("type")

        return value


class DateTime(DateTimeType[datetime.datetime]):
    datetime_pattern = re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
        r"[T ](?P<hour>\d{1,2}):(?P<minute>\d{1,2})"
        r"(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?"
        r"(?P<tzinfo>Z|[+-]\d{2}(?::?\d{2})?)?$"
    )
    datetime_type = datetime.datetime


class Time(DateTimeType[datetime.time]):
    datetime_pattern = re.compile(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{1,2})"
        r"(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?"
    )
    datetime_type = datetime.time


class Date(DateTimeType[datetime.date]):
    datetime_pattern = re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})$"
    )
    datetime_type = datetime.date


class UUID(Validator[uuid.UUID]):
    errors = {"type": "Must be a valid UUID."}

    def validate(self, value: object, allow_coerce: bool = False) -> uuid.UUID:
        if not isinstance(value, str):
            self.error("type")

        try:
            return uuid.UUID(value)
        except (TypeError, ValueError):
            self.error("type")


class Const(Validator[typing.Any]):
    errors = {"only_null": "Must be null.", "const": "Must be the value '{const}'."}

    def __init__(self, const: typing.Any):
        self.const = const

    def validate(self, value: object, allow_coerce: bool = False) -> typing.Any:
        if value != self.const:
            if self.const is None:
                self.error("only_null")
            self.error("const", const=self.const)
        return self.const


class _Any(Validator[typing.Any]):
    def validate(self, value: object, allow_coerce: bool = False) -> typing.Any:
        return value


Any = _Any()


class Union(Validator[typing.Any]):
    # variadic generics is not supported (
    errors = {"union": "Must match one of the union types."}

    def __init__(self, *items: Validator[typing.Any]):
        assert all(isinstance(k, Validator) for k in items)
        self.items = items

    def validate(self, value: object, allow_coerce: bool = False) -> typing.Any:
        errors = []
        for item in self.items:
            try:
                return item.validate(value, allow_coerce)
            except ValidationError as exc:
                errors.append(exc.detail)
                continue
        raise ValidationError(errors)


class Enumeration(typing.Protocol[_T_co]):
    def __getitem__(self, key: str) -> _T_co:
        ...  # pragma: nocover

    def __iter__(self) -> typing.Iterator[_T_co]:
        ...  # pragma: nocover


class Enum(Validator[_T]):
    errors = {"choice": "Must be one of {enum}.", "type": "Must be a string."}

    def __init__(self, items: Enumeration[_T]):
        self.items = items

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        if not isinstance(value, str):
            self.error("type")
        try:
            return self.items[value]
        except KeyError:
            enum = [str(getattr(x, "name", x)) for x in self.items]
            self.error("choice", enum=", ".join(enum))


class Choice(Validator[_T]):
    errors = {"choice": "Must be one of {choices}."}

    def __init__(self, items: typing.Collection[_T]):
        self.items = items

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        if value not in self.items:
            self.error("choice", choices=", ".join(repr(x) for x in self.items))
        return value  # type: ignore


class _MappingMixin(typing.Generic[_K, _V]):
    errors = {
        "type": "Must be an object.",
        "min_items": "Must have at least {count} items.",
        "max_items": "Must have no more than {count} items.",
    }

    def _validate(
        self: _ValidatorMixin,
        value: object,
        keys: typing.Optional[Validator[_K]],
        values: typing.Optional[Validator[_V]],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
        allow_coerce: bool = False,
    ) -> typing.Mapping[_K, _V]:
        if not isinstance(value, typing.Mapping):
            self.error("type")

        if min_items is not None and len(value) < min_items:
            self.error("min_items", count=min_items)

        elif max_items is not None and len(value) > max_items:
            self.error("max_items", count=max_items)

        validated: typing.Dict[_K, _V] = {}

        errors = {}

        for key, val in value.items():
            pos = key
            if keys is not None:
                try:
                    key = keys.validate(key)
                except ValidationError as exc:
                    errors[pos] = exc.detail
                    continue
            if values is not None:
                try:
                    val = values.validate(val, allow_coerce)
                except ValidationError as exc:
                    errors[pos] = exc.detail
                    continue
            validated[key] = val

        if errors:
            raise ValidationError(errors)

        return validated


class Mapping(_MappingMixin[_K, _V], Validator[typing.Mapping[_K, _V]]):
    def __init__(
        self,
        keys: Validator[_K],
        values: Validator[_V],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
    ):
        assert isinstance(keys, Validator)
        assert isinstance(values, Validator)
        assert min_items is None or isinstance(min_items, int)
        assert max_items is None or isinstance(max_items, int)
        self.keys = None if keys is Any else keys
        self.values = None if values is Any else values
        self.min_items = min_items
        self.max_items = max_items

    def validate(
        self, value: object, allow_coerce: bool = False
    ) -> typing.Mapping[_K, _V]:
        return self._validate(
            value, self.keys, self.values, self.min_items, self.max_items, allow_coerce
        )


class TypedMapping(_MappingMixin[_K, _V], Validator[_T]):
    def __init__(
        self,
        keys: Validator[_K],
        values: Validator[_V],
        converter: typing.Type[_T],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
    ):
        assert isinstance(keys, Validator)
        assert isinstance(values, Validator)
        assert min_items is None or isinstance(min_items, int)
        assert max_items is None or isinstance(max_items, int)
        assert callable(converter)

        self.keys = None if keys is Any else keys
        self.values = None if values is Any else values
        self.min_items = min_items
        self.max_items = max_items
        self.converter = converter

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        return self.converter(  # type: ignore
            self._validate(
                value,
                self.keys,
                self.values,
                self.min_items,
                self.max_items,
                allow_coerce,
            )
        )


class _ObjectMixin:
    errors = {
        "type": "Must be an object.",
        "invalid_key": "Object keys must be strings.",
        "required": 'The "{field_name}" field is required.',
    }

    def _validate(
        self: _ValidatorMixin,
        value: object,
        properties: typing.Mapping[str, Validator[typing.Any]],
        required: typing.Optional[typing.Tuple[str, ...]],
        allow_coerce: bool = False,
    ) -> typing.Mapping[str, typing.Any]:
        if not isinstance(value, typing.Mapping):
            self.error("type")

        validated: typing.Dict[str, typing.Any] = {}
        errors: typing.Dict[str, typing.Any] = {}

        for key in value.keys():
            if not isinstance(key, str):
                errors[key] = self.error_message("invalid_key")

        # Required properties
        if required:
            for key in required:
                if key not in value:
                    errors[key] = self.error_message("required", field_name=key)

        for key, child_schema in properties.items():
            if key not in value:
                continue
            item = value[key]
            try:
                validated[key] = child_schema.validate(item, allow_coerce)
            except ValidationError as exc:
                errors[key] = exc.detail

        if errors:
            raise ValidationError(errors)

        return validated


class Object(_ObjectMixin, Validator[typing.Mapping[str, typing.Any]]):
    def __init__(
        self,
        properties: typing.Mapping[str, Validator[typing.Any]],
        required: typing.Tuple[str, ...] = (),
    ):
        assert all(isinstance(k, str) for k in properties.keys())
        assert all(isinstance(v, Validator) for v in properties.values())
        assert isinstance(required, tuple) and all(isinstance(i, str) for i in required)
        self.properties = properties
        self.required = required

    def validate(
        self, value: object, allow_coerce: bool = False
    ) -> typing.Mapping[str, typing.Any]:
        return self._validate(value, self.properties, self.required, allow_coerce)


class TypedObject(_ObjectMixin, Validator[_T]):
    def __init__(
        self,
        properties: typing.Mapping[str, Validator[typing.Any]],
        converter: typing.Type[_T],
        required: typing.Tuple[str, ...] = (),
    ):
        assert all(isinstance(k, str) for k in properties.keys())
        assert all(isinstance(v, Validator) for v in properties.values())
        assert isinstance(required, tuple) and all(isinstance(i, str) for i in required)
        assert callable(converter)
        self.properties = properties
        self.required = required
        self.converter = converter

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        return self.converter(  # type: ignore
            **self._validate(value, self.properties, self.required, allow_coerce)
        )


class _ListMixin(typing.Generic[_T]):
    errors = {
        "type": "Must be an array.",
        "min_items": "Must have at least {count} items.",
        "max_items": "Must have no more than {count} items.",
        "unique_items": "This item is not unique.",
    }

    def _validate(
        self: _ValidatorMixin,
        value: object,
        items: typing.Optional[Validator[_T]],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
        unique_items: bool = False,
        allow_coerce: bool = False,
    ) -> typing.List[_T]:
        if not isinstance(value, list):
            self.error("type")

        if min_items is not None and len(value) < min_items:
            self.error("min_items", count=min_items)
        elif max_items is not None and len(value) > max_items:
            self.error("max_items", count=max_items)

        errors = {}
        validated = []

        if unique_items:
            seen_items = Uniqueness()

        for pos, item in enumerate(value):
            if items is not None:
                try:
                    item = items.validate(item, allow_coerce)
                except ValidationError as exc:
                    errors[pos] = exc.detail
                    continue

            if unique_items:
                # noinspection PyUnboundLocalVariable
                if item in seen_items:
                    self.error("unique_items")
                else:
                    seen_items.add(item)

            validated.append(item)

        if errors:
            raise ValidationError(errors)

        return validated


class List(_ListMixin[_V], Validator[typing.List[_V]]):
    def __init__(
        self,
        items: Validator[_V],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
        unique_items: bool = False,
    ):
        assert isinstance(items, Validator)
        assert min_items is None or isinstance(min_items, int)
        assert max_items is None or isinstance(max_items, int)
        assert isinstance(unique_items, bool)

        self.items = None if items is Any else items
        self.min_items = min_items
        self.max_items = max_items
        self.unique_items = unique_items

    def validate(self, value: object, allow_coerce: bool = False) -> typing.List[_V]:
        return self._validate(
            value,
            items=self.items,
            min_items=self.min_items,
            max_items=self.max_items,
            unique_items=self.unique_items,
            allow_coerce=allow_coerce,
        )


_T_Col = typing.TypeVar("_T_Col", bound=typing.Collection)  # type: ignore


class TypedList(_ListMixin[_V], Validator[_T]):
    def __init__(
        self,
        items: Validator[_V],
        converter: typing.Type[_T],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
        unique_items: bool = False,
    ):
        assert isinstance(items, Validator)
        assert min_items is None or isinstance(min_items, int)
        assert max_items is None or isinstance(max_items, int)
        assert callable(converter)

        self.items = None if items is Any else items
        self.min_items = min_items
        self.max_items = max_items
        self.unique_items = unique_items
        self.converter = converter

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        return self.converter(  # type: ignore
            self._validate(
                value,
                items=self.items,
                min_items=self.min_items,
                max_items=self.max_items,
                unique_items=self.unique_items,
                allow_coerce=allow_coerce,
            )
        )


class Tuple(TypedList[_V, typing.Tuple[_V, ...]]):
    def __init__(
        self,
        items: Validator[_V],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
        unique_items: bool = False,
    ):
        super().__init__(items, tuple, min_items, max_items, unique_items)


class Set(TypedList[_V, typing.Set[_V]]):
    def __init__(
        self,
        items: Validator[_V],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
    ):
        super().__init__(items, set, min_items, max_items, True)


class FrozenSet(TypedList[_V, typing.FrozenSet[_V]]):
    def __init__(
        self,
        items: Validator[_V],
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
    ):
        super().__init__(items, frozenset, min_items, max_items, True)


_T_Tup = typing.TypeVar("_T_Tup", bound=typing.Tuple)  # type: ignore


class TypedTuple(Validator[_T_Tup]):
    errors = {
        "type": "Must be an array.",
        "exact": "Must have exact {count} items.",
    }

    def __init__(
        self,
        items: typing.Tuple[Validator[typing.Any], ...],
        converter: typing.Optional[typing.Type[tuple]] = None,  # type: ignore
    ):
        assert isinstance(items, tuple) and all(
            isinstance(item, Validator) for item in items
        )
        assert converter is None or callable(converter)
        self.items = items
        self.converter = converter or tuple

    def validate(self, value: object, allow_coerce: bool = False) -> _T_Tup:
        if not isinstance(value, list):
            self.error("type")

        if len(value) != len(self.items):
            self.error("exact", count=len(self.items))

        errors = {}
        validated = []

        for pos, item in enumerate(value):
            try:
                validated.append(self.items[pos].validate(item, allow_coerce))
            except ValidationError as exc:
                errors[pos] = exc.detail

        if errors:
            raise ValidationError(errors)

        # noinspection PyArgumentList
        return self.converter(validated)  # type: ignore


class Uniqueness:
    """
    A set-like class that tests for uniqueness of primitive types.
    Ensures the `True` and `False` are treated as distinct from `1` and `0`,
    and coerces non-hashable instances that cannot be added to sets,
    into hashable representations that can.
    """

    TRUE = object()
    FALSE = object()

    def __init__(self) -> None:
        self._set: typing.Set[object] = set()

    def __contains__(self, item: object) -> bool:
        item = self.make_hashable(item)
        return item in self._set

    def add(self, item: object) -> None:
        item = self.make_hashable(item)
        self._set.add(item)

    def make_hashable(self, element: object) -> object:
        """
        Coerce a primitive into a uniquely hashable type, for uniqueness checks.
        """
        # Only primitive types can be handled.
        assert (element is None) or isinstance(
            element, (bool, int, float, str, list, dict)
        )

        if element is True:
            # Need to make `True` distinct from `1`.
            return self.TRUE

        elif element is False:
            # Need to make `False` distinct from `0`.
            return self.FALSE

        elif isinstance(element, list):
            # Represent lists using a two-tuple of ('list', (item, item, ...))
            return "list", tuple([self.make_hashable(item) for item in element])

        elif isinstance(element, dict):
            # Represent dicts using a two-tuple of ('dict', ((key, val), (key, val), ...))
            return (
                "dict",
                tuple(
                    [
                        (self.make_hashable(key), self.make_hashable(value))
                        for key, value in element.items()
                    ]
                ),
            )

        return element
