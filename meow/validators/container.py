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
    Object,
    UUID,
    Mapping,
    Array,
    Optional,
    Union,
)


def field(
    *,
    default=MISSING,
    default_factory=MISSING,
    repr=True,
    hash=None,
    init=True,
    compare=True,
    **kwargs,
):
    return _field(
        default=default,
        default_factory=default_factory,
        repr=repr,
        hash=hash,
        init=init,
        compare=compare,
        metadata=kwargs,
    )


class Container:
    _primitive_types: typing.Dict[typing.Any, typing.Type[Validator]] = {
        inspect.Parameter.empty: Any,
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

    _lookup_cache: dict
    _type_cache: dict

    def __init__(self, lookup_cache_size=5000, type_cache_size=1000):
        self._lookup_cache = {}
        self._type_cache = {}
        self._lookup_cache_size = lookup_cache_size
        self._type_cache_size = type_cache_size

    @classmethod
    def is_primitive_type(cls, tp: typing.Type) -> bool:
        return tp in cls._primitive_types

    @staticmethod
    def is_dataclass_type(tp: typing.Type) -> bool:
        return isinstance(tp, type) and is_dataclass(tp)

    def get_validator(self, tp: typing.Type) -> Validator:
        try:
            return self._lookup_cache[tp]
        except KeyError:
            pass
        validator = self._lookup_cache[tp] = self._make_validator(tp)
        if len(self._lookup_cache) > self._lookup_cache_size:  # pragma: nocover
            self._lookup_cache.pop(next(iter(self._lookup_cache)))
        return validator

    __getitem__ = get_validator

    def get_validator_parametrized(self, tp: typing.Type, **params) -> Validator:
        return self._make_validator(tp, params) if params else self.get_validator(tp)

    __call__ = get_validator_parametrized

    def _make_validator(self, tp: typing.Type, params: dict = None) -> Validator:
        cls, initial, optional = self._get_validator_constructor(tp)

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

    def _get_validator_constructor(self, tp: typing.Type):
        try:
            return self._type_cache[tp]
        except KeyError:
            pass
        if params := self._get_maybe_optional(tp):
            optional = True
        else:
            optional = False
            params = self._get_constructor_args(tp)
        cls, args = params
        if not self.is_dataclass_type(tp):
            self._type_cache[tp] = cls, args, optional
            if len(self._type_cache) > self._type_cache_size:  # pragma: nocover
                self._type_cache.pop(next(iter(self._type_cache)))
        return cls, args, optional

    def _get_maybe_optional(self, tp: typing.Type):
        if typing.get_origin(tp) is typing.Union:
            type_args = typing.get_args(tp)
            none_type = type(None)
            args = tuple(item for item in type_args if item is not none_type)  # type: ignore
            optional = len(args) != len(type_args)
            if optional:
                cls, params, _ = self._get_validator_constructor(
                    args[0] if len(args) == 1 else typing.Union.__getitem__(args)
                )
                return cls, params

    def _get_constructor_args(self, tp: typing.Type):
        if self.is_primitive_type(tp):
            return self._primitive_types[tp], None

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

        items: typing.Any

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

        raise TypeError("Don't know how to create validator for type %r" % tp)


V = Container()
get_validator = V.get_validator
get_validator_with_params = V.get_validator_parametrized
is_primitive_type = V.is_primitive_type
is_dataclass_type = V.is_dataclass_type
