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


class Validator(typing.Generic[_T]):

    errors: typing.Dict[str, str] = {}

    def error(self, code: str, **context: object) -> typing.NoReturn:
        raise ValidationError(self.error_message(code, **context))

    def error_message(self, code: str, **context: object) -> str:
        return self.errors[code].format_map(context)

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> _T:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__


class Optional(Validator[typing.Optional[_T]]):
    def __init__(self, validator: Validator[_T]):
        assert isinstance(validator, Validator)
        self.validator = validator

    def validate(
        self, value: typing.Any, allow_coerce: bool = False
    ) -> typing.Optional[_T]:
        if value is None:
            return None
        return self.validator.validate(value, allow_coerce)


class Adapter(Validator[_T], typing.Generic[_V, _T]):
    def __init__(
        self, validator: Validator[_V], converter: typing.Callable[[_V], _T],
    ):
        assert isinstance(validator, Validator)
        assert callable(converter)
        self.validator = validator
        self.converter = converter

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> _T:
        return self.converter(self.validator.validate(value, allow_coerce))


class Chain(Validator[_T], typing.Generic[_V, _T]):
    def __init__(
        self, validator: Validator[_V], next_validator: Validator[_T],
    ):
        assert isinstance(validator, Validator)
        assert isinstance(next_validator, Validator)
        self.validator = validator
        self.next_validator = next_validator

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> _T:
        return self.next_validator.validate(
            self.validator.validate(value, allow_coerce), allow_coerce
        )


class _Any(Validator[typing.Any]):
    def validate(self, value: typing.Any, allow_coerce: bool = False) -> typing.Any:
        return value


Any = _Any()


class String(Validator[str]):
    errors = {
        "type": "Must be a string.",
        "maxlength": "Must have no more than {maxlength} characters.",
        "minlength": "Must have at least {minlength} characters.",
        "pattern": "Must match the pattern /{pattern}/.",
    }

    def __init__(
        self,
        minlength: typing.Optional[int] = None,
        maxlength: typing.Optional[int] = None,
        pattern: typing.Optional[str] = None,
    ):

        assert maxlength is None or isinstance(maxlength, int)
        assert minlength is None or isinstance(minlength, int)
        assert pattern is None or isinstance(pattern, str)

        self.maxlength = maxlength
        self.minlength = minlength
        self.pattern = pattern

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> str:
        if not isinstance(value, str):
            self.error("type")

        if self.minlength is not None and len(value) < self.minlength:
            self.error("minlength", minlength=self.minlength)

        if self.maxlength is not None and len(value) > self.maxlength:
            self.error("maxlength", maxlength=self.maxlength)

        if self.pattern is not None and not re.search(self.pattern, value):
            self.error("pattern", pattern=self.pattern)

        return value


class Float(Validator[float]):
    errors = {
        "type": "Must be a number.",
        "minimum": "Must be greater than or equal to {minimum}.",
        "exclusive_minimum": "Must be greater than {minimum}.",
        "maximum": "Must be less than or equal to {maximum}.",
        "exclusive_maximum": "Must be less than {maximum}.",
    }

    def __init__(
        self,
        lt: typing.Optional[float] = None,
        gt: typing.Optional[float] = None,
        lte: typing.Optional[float] = None,
        gte: typing.Optional[float] = None,
    ):

        assert lt is None or isinstance(lt, (int, float))
        assert gt is None or isinstance(gt, (int, float))
        assert lte is None or isinstance(lte, (int, float))
        assert gte is None or isinstance(gte, (int, float))

        self.lt = lt
        self.gt = gt
        self.lte = lte
        self.gte = gte

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> float:
        if (
            value is None
            or isinstance(value, bool)
            or (not allow_coerce and not isinstance(value, (int, float)))
        ):
            self.error("type")

        try:
            result_value = float(value)
        except (TypeError, ValueError):
            self.error("type")

        if self.lt is not None:
            if result_value >= self.lt:
                self.error("exclusive_maximum", maximum=self.lt)
        elif self.lte is not None:
            if result_value > self.lte:
                self.error("maximum", maximum=self.lte)

        if self.gt is not None:
            if result_value <= self.gt:
                self.error("exclusive_minimum", minimum=self.gt)
        elif self.gte is not None:
            if result_value < self.gte:
                self.error("minimum", minimum=self.gte)

        return result_value


class Integer(Validator[int]):
    errors = {
        "type": "Must be an integer.",
        "minimum": "Must be greater than or equal to {minimum}.",
        "exclusive_minimum": "Must be greater than {minimum}.",
        "maximum": "Must be less than or equal to {maximum}.",
        "exclusive_maximum": "Must be less than {maximum}.",
    }

    def __init__(
        self,
        lt: typing.Optional[int] = None,
        gt: typing.Optional[int] = None,
        lte: typing.Optional[int] = None,
        gte: typing.Optional[int] = None,
    ):

        assert lt is None or isinstance(lt, int)
        assert gt is None or isinstance(gt, int)
        assert lte is None or isinstance(lte, int)
        assert gte is None or isinstance(gte, int)

        self.lt = lt
        self.gt = gt
        self.lte = lte
        self.gte = gte

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> int:
        if (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, float) and not value.is_integer())
            or (not allow_coerce and not isinstance(value, (int, float)))
        ):
            self.error("type")

        try:
            result_value = int(value)
        except (TypeError, ValueError):
            self.error("type")

        if self.lt is not None:
            if result_value >= self.lt:
                self.error("exclusive_maximum", maximum=self.lt)
        elif self.lte is not None:
            if result_value > self.lte:
                self.error("maximum", maximum=self.lte)

        if self.gt is not None:
            if result_value <= self.gt:
                self.error("exclusive_minimum", minimum=self.gt)
        elif self.gte is not None:
            if result_value < self.gte:
                self.error("minimum", minimum=self.gte)

        return result_value


class Boolean(Validator[bool]):
    errors = {"type": "Must be a boolean."}

    values: typing.Dict[str, bool] = {
        "on": True,
        "off": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if allow_coerce and isinstance(value, str):
            try:
                return self.values[value.lower()]
            except KeyError:
                self.error("type")

        self.error("type")


class _DateTimeType(Validator[_T]):
    datetime_pattern: typing.ClassVar[typing.Pattern[str]]
    datetime_type: typing.ClassVar[typing.Type[_T]]

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> _T:
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
            result_value = self.datetime_type(**kwargs)  # type: ignore
        except ValueError:
            self.error("type")

        return result_value


class DateTime(_DateTimeType[datetime.datetime]):
    errors = {"type": "Must be a datetime."}
    datetime_pattern = re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
        r"[T ](?P<hour>\d{1,2}):(?P<minute>\d{1,2})"
        r"(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?"
        r"(?P<tzinfo>Z|[+-]\d{2}(?::?\d{2})?)?$"
    )
    datetime_type = datetime.datetime


class Date(_DateTimeType[datetime.date]):
    errors = {"type": "Must be a date."}
    datetime_pattern = re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})$"
    )
    datetime_type = datetime.date


class Time(_DateTimeType[datetime.time]):
    errors = {"type": "Must be a time."}
    datetime_pattern = re.compile(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{1,2})"
        r"(?::(?P<second>\d{1,2})(?:\.(?P<microsecond>\d{1,6})\d{0,6})?)?"
    )
    datetime_type = datetime.time


class UUID(Validator[uuid.UUID]):
    errors = {"type": "Must be an uuid."}

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> uuid.UUID:
        if not isinstance(value, str):
            self.error("type")

        try:
            return uuid.UUID(value)
        except (TypeError, ValueError):
            self.error("type")


class Const(Validator[_T]):
    errors = {
        "only_null": "Must be a null.",
        "const": "Does not match {const!r}.",
    }

    def __init__(self, const: _T):
        self.const = const

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> _T:
        if value != self.const:
            if self.const is None:
                self.error("only_null")
            self.error("const", const=self.const)
        return self.const


class _Enumeration(typing.Protocol[_T_co]):
    def __getitem__(self, key: typing.Any) -> _T_co:
        ...  # pragma: nocover

    def __iter__(self) -> typing.Iterator[_T_co]:
        ...  # pragma: nocover


class Enum(Validator[_T]):
    errors = {"choice": "Does not match {choices}."}

    def __init__(self, items: _Enumeration[_T]):
        self.items = items

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> _T:
        try:
            return self.items[value]
        except (KeyError, ValueError, TypeError):
            # special case for python Enum
            enum = [str(getattr(x, "name", x)) for x in self.items]
            self.error("choice", choices=", ".join(enum))


class Choice(Validator[_T]):
    errors = {"choice": "Does not match {choices}."}

    def __init__(self, items: typing.Collection[_T]):
        self.items = items

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> _T:
        if value not in self.items:
            self.error("choice", choices=", ".join(repr(x) for x in self.items))
        return value  # type: ignore


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

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> typing.Any:
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

    def validate(self, value: typing.Any, allow_coerce: bool = False) -> typing.Any:
        try:
            self.cond.validate(value, allow_coerce)
        except ValidationError:
            if self.else_:
                return self.else_.validate(value, allow_coerce)
            raise
        else:
            return self.then.validate(value, allow_coerce)


class Mapping(Validator[typing.Dict[_K, _V]]):
    errors = {
        "type": "Must be an object.",
        "minitems": "Must have at least {minitems} properties.",
        "maxitems": "Must have no more then {maxitems} properties.",
    }

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
        self, value: typing.Any, allow_coerce: bool = False
    ) -> typing.Dict[_K, _V]:
        if not isinstance(value, typing.Mapping):
            self.error("type")

        if self.minitems is not None and len(value) < self.minitems:
            self.error("minitems", minitems=self.minitems)

        elif self.maxitems is not None and len(value) > self.maxitems:
            self.error("maxitems", maxitems=self.maxitems)

        errors = {}
        validated: typing.Dict[_K, _V] = {}
        for key, val in value.items():
            pos = key
            if self.keys is not None:
                try:
                    key = self.keys.validate(key)
                except ValidationError as exc:
                    errors[pos] = exc.detail
                    continue
            if self.values is not None:
                try:
                    val = self.values.validate(val, allow_coerce)
                except ValidationError as exc:
                    errors[pos] = exc.detail
                    continue
            validated[key] = val

        if errors:
            raise ValidationError(errors)

        return validated


class Router(Validator[typing.Dict[str, typing.Any]]):
    errors = {
        "type": "Must be an object.",
        "choice": "Does not match {choices}.",
        "minitems": "Must have at least {minitems} properties.",
        "maxitems": "Must have no more then {maxitems} properties.",
    }

    def __init__(
        self,
        items: typing.Dict[str, Validator[typing.Any]],
        minitems: typing.Optional[int] = None,
        maxitems: typing.Optional[int] = None,
    ):
        assert all(isinstance(k, str) for k in items.keys())
        assert all(isinstance(v, Validator) for v in items.values())
        assert minitems is None or isinstance(minitems, int)
        assert maxitems is None or isinstance(maxitems, int)
        self.items = items
        self.minitems = minitems
        self.maxitems = maxitems

    def validate(
        self, value: typing.Any, allow_coerce: bool = False
    ) -> typing.Dict[str, typing.Any]:
        if not isinstance(value, typing.Mapping):
            self.error("type")

        if self.minitems is not None and len(value) < self.minitems:
            self.error("minitems", minitems=self.minitems)

        elif self.maxitems is not None and len(value) > self.maxitems:
            self.error("maxitems", maxitems=self.maxitems)

        errors = {}
        validated: typing.Dict[str, typing.Any] = {}

        for key, val in value.items():
            item = self.items.get(key)
            if item is None:
                errors[key] = self.error_message(
                    "choice", choices=", ".join(self.items)
                )
                continue
            else:
                try:
                    val = item.validate(val, allow_coerce)
                except ValidationError as exc:
                    errors[key] = exc.detail
                    continue
            validated[key] = val

        if errors:
            raise ValidationError(errors)

        return validated


class Object(Validator[typing.Dict[str, typing.Any]]):
    errors = {
        "type": "Must be an object.",
        "invalid_key": "Object keys must be strings.",
        "required": "Required property is missing.",
    }

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
        self, value: typing.Any, allow_coerce: bool = False
    ) -> typing.Dict[str, typing.Any]:

        if not isinstance(value, typing.Mapping):
            self.error("type")

        validated: typing.Dict[str, typing.Any] = {}
        errors: typing.Dict[str, typing.Any] = {}

        for key in value.keys():
            if not isinstance(key, str):
                errors[key] = self.error_message("invalid_key")

        # Required properties
        if self.required:
            for key in self.required:
                if key not in value:
                    errors[key] = self.error_message("required")

        for key, child_schema in self.properties.items():
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


class List(Validator[typing.List[_T]]):
    errors = {
        "type": "Must be an array.",
        "minitems": "Must have at least {minitems} items.",
        "maxitems": "Must have no more than {maxitems} items.",
        "uniqueitems": "Non-unique array item.",
    }

    def __init__(
        self,
        items: Validator[_T],
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

    def validate(
        self, value: typing.Any, allow_coerce: bool = False
    ) -> typing.List[_T]:
        if not isinstance(value, list):
            self.error("type")

        if self.minitems is not None and len(value) < self.minitems:
            self.error("minitems", minitems=self.minitems)
        elif self.maxitems is not None and len(value) > self.maxitems:
            self.error("maxitems", maxitems=self.maxitems)

        errors = {}
        validated = []

        if self.uniqueitems:
            seen_items = _Uniqueness()

        for pos, item in enumerate(value):
            if self.items is not None:
                try:
                    item = self.items.validate(item, allow_coerce)
                except ValidationError as exc:
                    errors[pos] = exc.detail
                    continue

            if self.uniqueitems:
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


class TypedList(Validator[typing.Any]):
    errors = {
        "type": "Expected Array.",
        "minitems": "Must have at least {minitems} items.",
        "maxitems": "Must have no more than {maxitems} items.",
    }

    def __init__(self, *items: Validator[typing.Any]):
        assert all(isinstance(item, Validator) for item in items)
        self.items = items

    def validate(
        self, value: typing.Any, allow_coerce: bool = False
    ) -> typing.List[typing.Any]:
        if not isinstance(value, list):
            self.error("type")

        if len(value) != len(self.items):
            if len(value) < len(self.items):
                self.error("minitems", minitems=len(self.items))
            else:
                self.error("maxitems", maxitems=len(self.items))

        errors = {}
        validated = []

        for pos, item in enumerate(value):
            try:
                validated.append(self.items[pos].validate(item, allow_coerce))
            except ValidationError as exc:
                errors[pos] = exc.detail

        if errors:
            raise ValidationError(errors)

        return validated


class _Uniqueness:
    """
    A set-like class that tests for uniqueness of primitive types.
    Ensures the `True` and `False` are treated as distinct from `1` and `0`,
    and coerces non-hashable instances that cannot be added to sets,
    into hashable representations that can.
    """

    TRUE = object()
    FALSE = object()

    def __init__(self) -> None:
        self._set: typing.Set[typing.Any] = set()

    def __contains__(self, item: typing.Any) -> bool:
        item = self.make_hashable(item)
        return item in self._set

    def add(self, item: typing.Any) -> None:
        item = self.make_hashable(item)
        self._set.add(item)

    def make_hashable(self, element: typing.Any) -> typing.Any:
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
