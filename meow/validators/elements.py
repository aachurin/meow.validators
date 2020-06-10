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


import re
import datetime
import uuid
import typing
from .exception import ValidationError


class Validator:
    errors: typing.Dict[str, str] = {}

    def error(self, code: str, **context: object) -> typing.NoReturn:
        raise ValidationError(self.error_message(code, **context))

    def error_message(self, code: str, **context: object) -> str:
        return self.errors[code].format_map(context)

    def validate(self, value: object, allow_coerce: bool = False) -> object:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__


class Optional(Validator):
    def __init__(self, validator: Validator):
        self.validator = validator

    def validate(self, value: object, allow_coerce: bool = False) -> object:
        if value is None:
            return None
        return self.validator.validate(value, allow_coerce)


class Union(Validator):
    errors = {"union": "Must match one of the union types."}

    def __init__(self, items: typing.Sequence[Validator]):
        assert isinstance(items, typing.Sequence) and all(
            isinstance(k, Validator) for k in items
        )
        self.items = items

    def validate(self, value: object, allow_coerce: bool = False) -> object:
        for item in self.items:
            try:
                return item.validate(value, allow_coerce)
            except ValidationError:
                pass
        self.error("union")


class Enum(Validator):
    errors = {"choice": "Must be one of {enum}.", "type": "Must be a string."}

    def __init__(self, items: typing.Mapping[object, object]):
        self.items = items

    def validate(self, value: object, allow_coerce: bool = False) -> object:
        try:
            return self.items[value]
        except KeyError:
            enum = [str(getattr(x, "name", x)) for x in self.items]
            self.error("choice", enum=", ".join(enum))


class String(Validator):
    errors = {
        "type": "Must be a string.",
        "blank": "Must not be blank.",
        "max_length": "Must have no more than {max_length} characters.",
        "min_length": "Must have at least {min_length} characters.",
        "pattern": "Must match the pattern /{pattern}/.",
    }

    def __init__(
        self,
        max_length: typing.Optional[int] = None,
        min_length: typing.Optional[int] = None,
        pattern: typing.Optional[str] = None,
    ):

        assert max_length is None or isinstance(max_length, int)
        assert min_length is None or isinstance(min_length, int)
        assert pattern is None or isinstance(pattern, str)

        self.max_length = max_length
        self.min_length = min_length
        self.pattern = pattern

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


_T = typing.TypeVar("_T")


class NumericType(Validator, typing.Generic[_T]):
    errors = {
        "type": "Must be a number.",
        "integer": "Must be an integer.",
        "minimum": "Must be greater than or equal to {value}.",
        "exclusive_minimum": "Must be greater than {value}.",
        "maximum": "Must be less than or equal to {value}.",
        "exclusive_maximum": "Must be less than {value}.",
    }

    numeric_type: typing.Type[_T]

    def __init__(
        self,
        minimum: typing.Optional[_T] = None,
        maximum: typing.Optional[_T] = None,
        exclusive_minimum: bool = False,
        exclusive_maximum: bool = False,
    ):

        assert minimum is None or isinstance(minimum, (int, float))
        assert maximum is None or isinstance(maximum, (int, float))
        assert isinstance(exclusive_minimum, bool)
        assert isinstance(exclusive_maximum, bool)

        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum
        self.exclusive_maximum = exclusive_maximum

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


class Boolean(Validator):
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


class DateTimeType(Validator, typing.Generic[_T]):
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


class UUID(Validator):
    errors = {"type": "Must be a valid UUID."}

    def validate(self, value: object, allow_coerce: bool = False) -> uuid.UUID:
        if not isinstance(value, str):
            self.error("type")

        try:
            return uuid.UUID(value)
        except (TypeError, ValueError):
            self.error("type")


class Any(Validator):
    def validate(self, value: object, allow_coerce: bool = False) -> object:
        return value


class Cast(typing.Protocol):
    def __call__(self, **kwargs: object) -> object:
        raise NotImplementedError()


class Object(Validator):
    errors = {
        "type": "Must be an object.",
        "invalid_key": "Object keys must be strings.",
        "required": 'The "{field_name}" field is required.',
    }

    def __init__(
        self,
        properties: typing.Mapping[str, Validator],
        required: typing.Optional[typing.Sequence[str]] = None,
        cast: typing.Optional[Cast] = None,
    ):
        assert all(isinstance(k, str) for k in properties.keys())
        assert all(isinstance(v, Validator) for v in properties.values())
        assert required is None or (
            isinstance(required, typing.Sequence)
            and all(isinstance(i, str) for i in required)
        )
        assert cast is None or callable(cast)

        self.properties = properties
        self.required = required
        self.cast = cast

    def validate(self, value: object, allow_coerce: bool = False) -> object:
        if not isinstance(value, dict):
            self.error("type")

        validated = {}

        # Ensure all property keys are strings.
        errors: typing.Dict[str, object] = {}

        for key in value.keys():
            if not isinstance(key, str):
                errors[key] = self.error_message("invalid_key")

        # Required properties
        if self.required:
            for key in self.required:
                if key not in value:
                    errors[key] = self.error_message("required", field_name=key)

        # Properties
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

        if self.cast:
            return self.cast(**validated)

        return validated  # pragma: nocover


class Mapping(Validator):
    errors = {
        "type": "Must be an object.",
        "min_items": "Must have at least {count} items.",
        "max_items": "Must have no more than {count} items.",
    }

    def __init__(
        self,
        keys: typing.Optional[Validator] = None,
        values: typing.Optional[Validator] = None,
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
        cast: typing.Optional[
            typing.Callable[
                [typing.Mapping[object, object]], typing.Mapping[object, object]
            ]
        ] = None,
    ):

        assert keys is None or isinstance(keys, Validator)
        assert values is None or isinstance(values, Validator)
        assert min_items is None or isinstance(min_items, int)
        assert max_items is None or isinstance(max_items, int)
        assert cast is None or callable(cast)

        self.keys = keys
        self.values = values
        self.min_items = min_items
        self.max_items = max_items
        self.cast = cast

    def validate(
        self, value: object, allow_coerce: bool = False
    ) -> typing.Mapping[object, object]:
        if not isinstance(value, dict):
            self.error("type")

        if self.min_items is not None and len(value) < self.min_items:
            self.error("min_items", count=self.min_items)

        elif self.max_items is not None and len(value) > self.max_items:
            self.error("max_items", count=self.max_items)

        validated = {}

        errors = {}
        keys = self.keys
        values = self.values

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

        if self.cast:
            return self.cast(validated)

        return validated


class Array(Validator):
    errors = {
        "type": "Must be an array.",
        "min_items": "Must have at least {count} items.",
        "max_items": "Must have no more than {count} items.",
        "unique_items": "This item is not unique.",
    }

    def __init__(
        self,
        items: typing.Union[None, Validator, typing.Sequence[Validator]] = None,
        min_items: typing.Optional[int] = None,
        max_items: typing.Optional[int] = None,
        unique_items: bool = False,
        cast: typing.Optional[
            typing.Callable[[typing.Sequence[object]], typing.Sequence[object]]
        ] = None,
    ):
        assert (
            items is None
            or isinstance(items, Validator)
            or (
                isinstance(items, typing.Sequence)
                and all(isinstance(i, Validator) for i in items)
            )
        )
        assert min_items is None or isinstance(min_items, int)
        assert max_items is None or isinstance(max_items, int)
        assert isinstance(unique_items, bool)
        assert cast is None or callable(cast)

        self.unique_items = unique_items
        self.cast = cast
        self.items = items
        self.min_items = min_items
        self.max_items = max_items
        if isinstance(items, typing.Sequence):
            self.min_items = self.max_items = len(items)

    def validate(
        self, value: object, allow_coerce: bool = False
    ) -> typing.Sequence[object]:
        if not isinstance(value, list):
            self.error("type")

        if self.min_items is not None and len(value) < self.min_items:
            self.error("min_items", count=self.min_items)
        elif self.max_items is not None and len(value) > self.max_items:
            self.error("max_items", count=self.max_items)

        errors = {}
        if self.unique_items:
            seen_items = Uniqueness()

        if isinstance(self.items, typing.Sequence):
            indexed: typing.Optional[typing.Sequence[Validator]] = self.items
            validator = None
        else:
            indexed = None
            validator = self.items

        validated = []
        for pos, item in enumerate(value):
            try:
                if indexed is not None:
                    item = indexed[pos].validate(item, allow_coerce)
                elif validator is not None:
                    # noinspection PyUnresolvedReferences
                    item = validator.validate(item, allow_coerce)

                if self.unique_items:
                    # noinspection PyUnboundLocalVariable
                    if item in seen_items:
                        self.error("unique_items")
                    else:
                        seen_items.add(item)

                validated.append(item)

            except ValidationError as exc:
                errors[pos] = exc.detail

        if errors:
            raise ValidationError(errors)

        if self.cast:
            return self.cast(validated)

        return validated


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
