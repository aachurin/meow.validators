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
        "type": "Expected String.",
        "maxlength": "String {value!r} exceeds maximum length of {maxlength}.",
        "minlength": "String {value!r} is less than minimum length of {minlength}.",
        "pattern": "String {value!r} does not match regex pattern /{pattern}/.",
    }

    def __init__(
        self,
        minlength: typing.Optional[object] = None,
        maxlength: typing.Optional[object] = None,
        pattern: typing.Optional[object] = None,
    ):

        assert maxlength is None or isinstance(maxlength, int)
        assert minlength is None or isinstance(minlength, int)
        assert pattern is None or isinstance(pattern, str)

        self.maxlength: typing.Optional[int] = maxlength
        self.minlength: typing.Optional[int] = minlength
        self.pattern: typing.Optional[str] = pattern

    def validate(self, value: object, allow_coerce: bool = False) -> str:
        if not isinstance(value, str):
            self.error("type")

        if self.minlength is not None and len(value) < self.minlength:
            self.error("minlength", value=value, minlength=self.minlength)

        if self.maxlength is not None and len(value) > self.maxlength:
            self.error("maxlength", value=value, maxlength=self.maxlength)

        if self.pattern is not None and not re.search(self.pattern, value):
            self.error("pattern", value=value, pattern=self.pattern)

        return value


class NumericType(Validator[_T]):
    numeric_type: typing.ClassVar[typing.Type[_T]]

    def __init__(
        self,
        lt: typing.Optional[_T] = None,
        gt: typing.Optional[_T] = None,
        lte: typing.Optional[_T] = None,
        gte: typing.Optional[_T] = None,
    ):

        assert lt is None or isinstance(lt, (int, float))
        assert gt is None or isinstance(gt, (int, float))
        assert lte is None or isinstance(lte, (int, float))
        assert gte is None or isinstance(gte, (int, float))

        self.lt = lt
        self.gt = gt
        self.lte = lte
        self.gte = gte

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        if (
            self.numeric_type is int
            and isinstance(value, float)
            and not value.is_integer()
        ):
            self.error("type")
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

        if self.lt is not None:
            if not (value < self.lt):
                if value == self.lt:
                    self.error("equal_max", value=value, max=self.lt)
                self.error("max", value=value, max=self.lt)
        elif self.lte is not None:
            if not (value <= self.lte):
                self.error("max", value=value, max=self.lte)

        if self.gt is not None:
            if not (value > self.gt):
                if value == self.gt:
                    self.error("equal_min", value=value, min=self.gt)
                self.error("min", value=value, min=self.gt)
        elif self.gte is not None:
            if not (value >= self.gte):
                self.error("min", value=value, min=self.gte)

        return value


class Float(NumericType[float]):
    errors = {
        "type": "Expected Float.",
        "min": "Float {value} is less then minimum value of {min}.",
        "equal_min": "Float {value} equals minimum value of {min}.",
        "max": "Float {value} exceeds maximum value of {max}.",
        "equal_max": "Float {value} equals maximum value of {max}.",
    }
    numeric_type = float


class Integer(NumericType[int]):
    errors = {
        "type": "Expected Integer.",
        "min": "Integer {value} is less then minimum value of {min}.",
        "equal_min": "Integer {value} equals minimum value of {min}.",
        "max": "Integer {value} exceeds maximum value of {max}.",
        "equal_max": "Integer {value} equals maximum value of {max}.",
    }
    numeric_type = int


class Boolean(Validator[bool]):
    errors = {"type": "Expected Boolean."}

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
    errors = {"type": "Expected DateTime."}
    datetime_pattern = re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
        r"[T ](?P<hour>\d{1,2}):(?P<minute>\d{1,2})"
        r"(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?"
        r"(?P<tzinfo>Z|[+-]\d{2}(?::?\d{2})?)?$"
    )
    datetime_type = datetime.datetime


class Time(DateTimeType[datetime.time]):
    errors = {"type": "Expected Time."}
    datetime_pattern = re.compile(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{1,2})"
        r"(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?"
    )
    datetime_type = datetime.time


class Date(DateTimeType[datetime.date]):
    errors = {"type": "Expected Date."}
    datetime_pattern = re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})$"
    )
    datetime_type = datetime.date


class UUID(Validator[uuid.UUID]):
    errors = {"type": "Expected UUID."}

    def validate(self, value: object, allow_coerce: bool = False) -> uuid.UUID:
        if not isinstance(value, str):
            self.error("type")

        try:
            return uuid.UUID(value)
        except (TypeError, ValueError):
            self.error("type")


class Const(Validator[typing.Any]):
    errors = {
        "only_null": "Value does not match null.",
        "const": "Value does not match {const!r}.",
    }

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
    def __init__(self, *items: Validator[typing.Any]):
        union_items: typing.List[Validator[typing.Any]] = []
        for item in items:
            if isinstance(item, Union):
                union_items.extend(item.items)
            else:
                assert isinstance(item, Validator)
                union_items.append(item)
        self.items: typing.Tuple[Validator[typing.Any], ...] = tuple(union_items)

    def validate(self, value: object, allow_coerce: bool = False) -> typing.Any:
        errors = []
        for item in self.items:
            try:
                return item.validate(value, allow_coerce)
            except ValidationError as exc:
                if exc.detail not in errors:
                    errors.append(exc.detail)
                continue
        raise ValidationError({"_union": errors})


class If(Validator[typing.Any]):
    def __init__(
        self,
        cond: Validator[typing.Any],
        then: Validator[typing.Any],
        else_: typing.Optional[Validator[typing.Any]] = None,
    ):
        self.cond = cond
        self.then = then
        self.else_ = else_

    def validate(self, value: object, allow_coerce: bool = False) -> typing.Any:
        try:
            self.cond.validate(value, allow_coerce)
        except ValidationError:
            if self.else_:
                return self.else_.validate(value, allow_coerce)
            raise
        else:
            return self.then.validate(value, allow_coerce)


class Enumeration(typing.Protocol[_T_co]):
    def __getitem__(self, key: typing.Any) -> _T_co:
        ...  # pragma: nocover

    def __iter__(self) -> typing.Iterator[_T_co]:
        ...  # pragma: nocover


class Enum(Validator[_T]):
    errors = {
        "choice": "Value {value!r} does not match any of the available choices: {choices}."
    }

    def __init__(self, items: Enumeration[_T]):
        self.items = items

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        try:
            return self.items[value]
        except (KeyError, ValueError, TypeError):
            enum = [str(getattr(x, "name", x)) for x in self.items]
            self.error("choice", value=value, choices=", ".join(enum))


class Choice(Validator[_T]):
    errors = {
        "choice": "Value {value!r} does not match any of the available choices: {choices}."
    }

    def __init__(self, items: typing.Collection[_T]):
        self.items = items

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        if value not in self.items:
            self.error(
                "choice", value=value, choices=", ".join(repr(x) for x in self.items)
            )
        return value  # type: ignore


class _MappingMixin(typing.Generic[_K, _V]):
    errors = {
        "type": "Expected Object.",
        "minitems": "Object property count {count} is less then minimum count of {minitems}.",
        "maxitems": "Object property count {count} exceeds maximum count of {maxitems}.",
    }

    def _validate(
        self: _ValidatorMixin,
        value: object,
        keys: typing.Optional[Validator[_K]],
        values: typing.Optional[Validator[_V]],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
        allow_coerce: bool = False,
    ) -> typing.Mapping[_K, _V]:
        if not isinstance(value, typing.Mapping):
            self.error("type")

        if minitems is not None and len(value) < minitems:
            self.error("minitems", count=len(value), minitems=minitems)

        elif maxitems is not None and len(value) > maxitems:
            self.error("maxitems", count=len(value), maxitems=maxitems)

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
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
    ):
        assert isinstance(keys, Validator)
        assert isinstance(values, Validator)
        assert minitems is None or isinstance(minitems, int)
        assert maxitems is None or isinstance(maxitems, int)
        self.keys = None if keys is Any else keys
        self.values = None if values is Any else values
        self.minitems = minitems
        self.maxitems = maxitems

    def validate(
        self, value: object, allow_coerce: bool = False
    ) -> typing.Mapping[_K, _V]:
        return self._validate(
            value, self.keys, self.values, self.minitems, self.maxitems, allow_coerce
        )


class TypedMapping(_MappingMixin[_K, _V], Validator[_T]):
    def __init__(
        self,
        keys: Validator[_K],
        values: Validator[_V],
        converter: typing.Type[_T],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
    ):
        assert isinstance(keys, Validator)
        assert isinstance(values, Validator)
        assert minitems is None or isinstance(minitems, int)
        assert maxitems is None or isinstance(maxitems, int)
        assert callable(converter)

        self.keys = None if keys is Any else keys
        self.values = None if values is Any else values
        self.minitems = minitems
        self.maxitems = maxitems
        self.converter = converter

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        return self.converter(  # type: ignore
            self._validate(
                value,
                self.keys,
                self.values,
                self.minitems,
                self.maxitems,
                allow_coerce,
            )
        )


class _ObjectMixin:
    errors = {
        "type": "Expected Object.",
        "invalid_key": "Object keys must be strings.",
        "required": "Required property is missing.",
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
                    errors[key] = self.error_message("required")

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
        required: typing.Optional[typing.Tuple[str, ...]] = None,
    ):
        assert all(isinstance(k, str) for k in properties.keys())
        assert all(isinstance(v, Validator) for v in properties.values())
        assert required is None or (
            isinstance(required, tuple) and all(isinstance(i, str) for i in required)
        )
        self.properties = properties
        self.required = tuple(properties.keys()) if required is None else required

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
        "type": "Expected Array.",
        "minitems": "Array item count {count} is less than minimum count of {minitems}.",
        "maxitems": "Array item count {count} exceeds maximum count of {maxitems}.",
        "uniqueitems": "Non-unique array item.",
    }

    def _validate(
        self: _ValidatorMixin,
        value: object,
        items: typing.Optional[Validator[_T]],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
        uniqueitems: bool = False,
        allow_coerce: bool = False,
    ) -> typing.List[_T]:
        if not isinstance(value, list):
            self.error("type")

        if minitems is not None and len(value) < minitems:
            self.error("minitems", count=len(value), minitems=minitems)
        elif maxitems is not None and len(value) > maxitems:
            self.error("maxitems", count=len(value), maxitems=maxitems)

        errors = {}
        validated = []

        if uniqueitems:
            seen_items = Uniqueness()

        for pos, item in enumerate(value):
            if items is not None:
                try:
                    item = items.validate(item, allow_coerce)
                except ValidationError as exc:
                    errors[pos] = exc.detail
                    continue

            if uniqueitems:
                # noinspection PyUnboundLocalVariable
                if item in seen_items:
                    errors[pos] = self.error_message("uniqueitems")
                    continue
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
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
        uniqueitems: bool = False,
    ):
        assert isinstance(items, Validator)
        assert minitems is None or isinstance(minitems, int)
        assert maxitems is None or isinstance(maxitems, int)
        assert isinstance(uniqueitems, bool)

        self.items = None if items is Any else items
        self.minitems = minitems
        self.maxitems = maxitems
        self.uniqueitems = uniqueitems

    def validate(self, value: object, allow_coerce: bool = False) -> typing.List[_V]:
        return self._validate(
            value,
            items=self.items,
            minitems=self.minitems,
            maxitems=self.maxitems,
            uniqueitems=self.uniqueitems,
            allow_coerce=allow_coerce,
        )


_T_Col = typing.TypeVar("_T_Col", bound=typing.Collection)  # type: ignore


class TypedList(_ListMixin[_V], Validator[_T]):
    def __init__(
        self,
        items: Validator[_V],
        converter: typing.Type[_T],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
        uniqueitems: bool = False,
    ):
        assert isinstance(items, Validator)
        assert minitems is None or isinstance(minitems, int)
        assert maxitems is None or isinstance(maxitems, int)
        assert callable(converter)

        self.items = None if items is Any else items
        self.minitems = minitems
        self.maxitems = maxitems
        self.uniqueitems = uniqueitems
        self.converter = converter

    def validate(self, value: object, allow_coerce: bool = False) -> _T:
        return self.converter(  # type: ignore
            self._validate(
                value,
                items=self.items,
                minitems=self.minitems,
                maxitems=self.maxitems,
                uniqueitems=self.uniqueitems,
                allow_coerce=allow_coerce,
            )
        )


class Tuple(TypedList[_V, typing.Tuple[_V, ...]]):
    def __init__(
        self,
        items: Validator[_V],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
        uniqueitems: bool = False,
    ):
        super().__init__(items, tuple, minitems, maxitems, uniqueitems)


class Set(TypedList[_V, typing.Set[_V]]):
    def __init__(
        self,
        items: Validator[_V],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
    ):
        super().__init__(items, set, minitems, maxitems, True)


class FrozenSet(TypedList[_V, typing.FrozenSet[_V]]):
    def __init__(
        self,
        items: Validator[_V],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
    ):
        super().__init__(items, frozenset, minitems, maxitems, True)


_T_Tup = typing.TypeVar("_T_Tup", bound=typing.Tuple)  # type: ignore


class TypedTuple(Validator[_T_Tup]):
    errors = {
        "type": "Expected Array.",
        "minitems": "Array item count {count} is less than minimum count of {minitems}.",
        "maxitems": "Array item count {count} exceeds maximum count of {maxitems}.",
    }

    def __init__(
        self,
        *items: Validator[typing.Any],
        converter: typing.Optional[typing.Type[tuple]] = None,  # type: ignore
    ):
        assert all(isinstance(item, Validator) for item in items)
        assert converter is None or callable(converter)
        self.items = items
        self.converter = converter or tuple

    def validate(self, value: object, allow_coerce: bool = False) -> _T_Tup:
        if not isinstance(value, list):
            self.error("type")

        if len(value) != len(self.items):
            if len(value) < len(self.items):
                self.error("minitems", count=len(value), minitems=len(self.items))
            else:
                self.error("maxitems", count=len(value), maxitems=len(self.items))

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
