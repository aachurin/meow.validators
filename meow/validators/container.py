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


import inspect
import datetime
import uuid
import typing
import enum
from dataclasses import field as _field, fields, is_dataclass, MISSING
from .elements import (
    Validator,
    Any,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Time,
    Date,
    UUID,
    Enum,
    Object,
    Mapping,
    Array,
    Optional,
    Union,
)


def field(  # type: ignore
    *,
    default=MISSING,
    default_factory=MISSING,
    repr=True,
    hash=None,
    init=True,
    compare=True,
    **kwargs,
):
    return _field(  # type: ignore
        default=default,
        default_factory=default_factory,
        repr=repr,
        hash=hash,
        init=init,
        compare=compare,
        metadata=kwargs,
    )


CachedConstructor = typing.Tuple[
    typing.Type[Validator], typing.Optional[typing.Mapping[str, object]], bool
]
Constructor = typing.Tuple[
    typing.Type[Validator], typing.Optional[typing.Mapping[str, object]]
]


class Container:
    _primitive_types = {
        typing.Any: Any,
        str: String,
        float: Float,
        int: Integer,
        bool: Boolean,
        datetime.datetime: DateTime,
        datetime.time: Time,
        datetime.date: Date,
        uuid.UUID: UUID,
    }

    _lookup_cache: typing.Dict[typing.Type[object], Validator]
    _type_cache: typing.Dict[typing.Type[object], CachedConstructor]

    def __init__(
        self,
        lookup_cache_size: int = 5000,
        type_cache_size: int = 2500,
        default: typing.Optional[
            typing.Callable[[typing.Type[object]], Constructor]
        ] = None,
    ):
        self._lookup_cache = {}
        self._type_cache = {}
        self._lookup_cache_size = lookup_cache_size
        self._type_cache_size = type_cache_size
        self._default = default

    @classmethod
    def is_primitive_type(cls, tp: typing.Type[object]) -> bool:
        return tp in cls._primitive_types

    @staticmethod
    def is_dataclass_type(tp: typing.Type[object]) -> bool:
        return isinstance(tp, type) and is_dataclass(tp)

    @staticmethod
    def is_enum_type(tp: typing.Type[object]) -> bool:
        return isinstance(tp, type) and issubclass(tp, enum.Enum)

    def get_validator(self, tp: typing.Type[object]) -> Validator:
        try:
            return self._lookup_cache[tp]
        except KeyError:
            pass
        validator = self._lookup_cache[tp] = self._make_validator(tp)
        if len(self._lookup_cache) > self._lookup_cache_size:  # pragma: nocover
            self._lookup_cache.pop(next(iter(self._lookup_cache)))
        return validator

    __getitem__ = get_validator

    def get_validator_parametrized(
        self, tp: typing.Type[object], **params: object
    ) -> Validator:
        return self._make_validator(tp, params) if params else self.get_validator(tp)

    __call__ = get_validator_parametrized

    def _make_validator(
        self,
        tp: typing.Type[object],
        params: typing.Optional[typing.Mapping[str, object]] = None,
    ) -> Validator:
        cls, initial, optional = self._get_constructor(tp)

        if params and initial:
            # noinspection PyArgumentList
            ret = cls(**{**initial, **params})  # type: ignore
        elif params:
            # noinspection PyArgumentList
            ret = cls(**params)  # type: ignore
        elif initial:
            # noinspection PyArgumentList
            ret = cls(**initial)  # type: ignore
        else:
            ret = cls()

        if optional:
            return Optional(ret)
        return ret

    def _get_constructor(self, tp: typing.Type[object]) -> CachedConstructor:
        try:
            return self._type_cache[tp]
        except KeyError:
            pass
        if constructor := self._get_maybe_optional(tp):
            optional = True
        else:
            optional = False
            constructor = self._make_constructor(tp)
        cls, kwargs = constructor
        self._type_cache[tp] = cls, kwargs, optional
        if len(self._type_cache) > self._type_cache_size:  # pragma: nocover
            self._type_cache.pop(next(iter(self._type_cache)))
        return cls, kwargs, optional

    def _get_maybe_optional(
        self, tp: typing.Type[object]
    ) -> typing.Optional[Constructor]:
        if typing.get_origin(tp) is typing.Union:
            type_args = typing.get_args(tp)
            none_type = type(None)
            args = tuple(item for item in type_args if item is not none_type)  # type: ignore
            optional = len(args) != len(type_args)
            if optional:
                cls, kwargs, _ = self._get_constructor(
                    args[0] if len(args) == 1 else typing.Union.__getitem__(args)
                )
                return cls, kwargs
        return None

    def _make_constructor(self, tp: typing.Type[object]) -> Constructor:
        if self.is_primitive_type(tp):
            return self._primitive_types[tp], None

        if self.is_enum_type(tp):
            return Enum, {"items": tp}

        if self.is_dataclass_type(tp):
            properties = {}
            required = []
            # noinspection PyDataclass
            for fld in fields(tp):
                if not fld.init:
                    continue
                if fld.default is MISSING and fld.default_factory is MISSING:  # type: ignore
                    required.append(fld.name)
                properties[fld.name] = self.get_validator_parametrized(
                    fld.type, **fld.metadata
                )
            return Object, {"properties": properties, "required": required, "cast": tp}

        origin = typing.get_origin(tp) or tp
        type_args = typing.get_args(tp)

        items: typing.Optional[object]

        if origin is typing.Union:
            items = [self.get_validator(arg) for arg in type_args]
            return Union, {"items": items}

        if origin is tuple:
            if not type_args:
                items = None
            elif type_args[-1] is ...:
                items = self.get_validator(type_args[0])
            else:
                items = [self.get_validator(arg) for arg in type_args]
            return Array, {"items": items, "cast": tuple}

        if issubclass(origin, typing.Sequence):
            if inspect.isabstract(origin) or origin is list:
                # default cast is list
                cast = None
            else:
                cast = origin
            items = self.get_validator(type_args[0]) if type_args else None
            return Array, {"items": items, "cast": cast}

        if issubclass(origin, typing.AbstractSet):
            if inspect.isabstract(origin):
                cast = set
            else:
                cast = origin
            items = self.get_validator(type_args[0]) if type_args else None
            return Array, {"items": items, "cast": cast, "unique_items": True}

        if issubclass(origin, typing.Mapping):
            if inspect.isabstract(origin) or origin is dict:
                # default cast is dict
                cast = None
            else:
                cast = origin
            keys = self.get_validator(type_args[0]) if type_args else None
            values = self.get_validator(type_args[1]) if type_args else None
            return Mapping, {"keys": keys, "values": values, "cast": cast}

        if self._default:
            return self._default(tp)

        raise TypeError("Don't know how to create validator for type %r" % tp)


V = Container()
get_validator = V.get_validator
get_validator_with_params = V.get_validator_parametrized
is_primitive_type = V.is_primitive_type
is_dataclass_type = V.is_dataclass_type
