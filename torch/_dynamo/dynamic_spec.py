"""Dynamic shape specification types for ``torch.compile`` and ``torch.export``.

Provides class `IntSpec` for fine-grained control over whether an integer
(dimension size or scalar argument) is treated as static, backed, or unbacked
during compilation.

Backed vs. unbacked
-------------------
``torch.compile`` provides two kinds of dynamic shapes: ``backed`` and
``unbacked``. ``torch.compile`` guards on ``backed`` dynamic shapes and does
not provide a guarantee that no guards will be added to them. User code,
dynamo, inductor, and autograd all can add guards when tracing through
branching, e.g. ``if x.size() > 10``. Moreover, for 0/1 specializations,
backed symbols are specialized unconditionally to ``0``, ``1``, or ``>=2``
even without encountering a branching on those ranges.

On the contrary, ``unbacked`` dynamic shapes are guaranteed not to be guarded
on and are not 0/1 specialized. However, there is a possibility of throwing a
data-dependent error when a branch that requires their value is encountered
and no explicit unbacked handling is defined. The framework is converging to
a state where it won't throw DDE but rather pick general paths. One downside
of using unbacked is missed optimization opportunities due to either perf
bugs or picking general paths, or using a fixed non-example input-based hint.
An example of picking general paths is assuming input not contiguous in
functions called ``contiguous()`` and ``reshape()`` when it cannot be
symbolically proven, with a change of introducing a clone.

For more info see
https://dev-discuss.pytorch.org/t/backed-to-unbacked-from-guardable-to-guardless-shapes-in-pytorch/3333.
"""

import enum
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any, ClassVar

import torch
import torch.utils._pytree as pytree


__all__ = [
    "IntSpecType",
    "IntSpec",
    "TensorSpec",
    "ObjectSpec",
    "get_active_spec_for_arg",
    "get_active_spec_for_dim",
]


class IntSpecType(enum.Enum):
    """How an integer should be treated during compilation.

    STATIC: compile-time constant; triggers recompilation if the value changes.
    BACKED: symbolic with guards and 0/1 specialization permitted.
    UNBACKED: symbolic, no guards, no 0/1 specialization; may raise a data-dependent error on branching.
    """

    STATIC = "static"
    BACKED = "backed"
    UNBACKED = "unbacked"


class IntSpec:
    """Shape specification for a single integer (dimension size or scalar arg).

    Create via a classmethod factory or the constructor directly:

        IntSpec.static("x", value=10)
        IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        IntSpec("x", IntSpecType.STATIC, value=10)

    ``type`` is fixed at construction; all other fields are mutable via
    fluent setters that double as getters (no arg = read, one arg = write):

        spec = IntSpec.backed("batch", min=1, max=64)
        spec.guarding_hint(32)   # set, returns self
        spec.guarding_hint()     # get, returns 32
        spec.min(1).max(64)      # chain
    """

    _name: str | None
    _type: IntSpecType
    _value: int | None
    _min: int | None
    _max: int | None
    _guarding_hint: int | None
    _optimization_hint: int | None

    __slots__ = (
        "_name",
        "_type",
        "_value",
        "_min",
        "_max",
        "_guarding_hint",
        "_optimization_hint",
    )

    def __init__(
        self,
        name: str | None,
        type: IntSpecType,
        *,
        min: int | None = None,
        max: int | None = None,
        value: int | None = None,
        guarding_hint: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if not isinstance(type, IntSpecType):
            raise TypeError(f"IntSpec type must be an IntSpecType, got {type!r}")
        self._type = type
        self._name = name
        self._min = min
        self._max = max
        self._value = value
        self._guarding_hint = guarding_hint
        self._optimization_hint = optimization_hint
        self._validate()

    def __setattr__(self, key: str, value: Any) -> None:
        # ``_type`` is the only pinned slot — it drives the per-mode
        # validation rules and integration-level dispatch (BACKED vs.
        # UNBACKED).
        if key == "_type" and hasattr(self, "_type"):
            raise AttributeError("IntSpec type is immutable; cannot reassign")
        object.__setattr__(self, key, value)

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"IntSpec attribute {key!r} cannot be deleted")

    _MODE_KWARG_HINT: ClassVar[dict[IntSpecType, tuple[str, str]]] = {
        IntSpecType.STATIC: ("static", "value"),
        IntSpecType.BACKED: ("backed", "guarding_hint"),
        IntSpecType.UNBACKED: ("unbacked", "optimization_hint"),
    }

    @staticmethod
    def _check_name(value: Any, type_: IntSpecType) -> None:
        if value is not None and not isinstance(value, str):
            factory, kwarg = IntSpec._MODE_KWARG_HINT[type_]
            raise TypeError(
                f"IntSpec.name must be str or None, got "
                f"{value.__class__.__name__}; if you meant to pass a "
                f"value/hint, use a keyword argument "
                f"(e.g. IntSpec.{factory}({kwarg}={value!r}))"
            )

    @staticmethod
    def _check_int_field(field_name: str, value: Any) -> None:
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise TypeError(
                f"IntSpec.{field_name} must be int or None, got "
                f"{value.__class__.__name__}"
            )

    # -- validation --------------------------------------------------------
    #
    # Single entry point: type checks (name, int fields), per-mode rules,
    # and cross-field invariants like ``min <= max``. Run on every
    # construction and on every fluent set (via ``_try_set``).

    def _validate(self) -> None:
        IntSpec._check_name(self._name, self._type)
        IntSpec._check_int_field("min", self._min)
        IntSpec._check_int_field("max", self._max)
        IntSpec._check_int_field("value", self._value)
        IntSpec._check_int_field("guarding_hint", self._guarding_hint)
        IntSpec._check_int_field("optimization_hint", self._optimization_hint)
        if self._type is IntSpecType.STATIC:
            if self._min is not None or self._max is not None:
                raise ValueError(
                    "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC"
                )
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
        elif self._type is IntSpecType.BACKED:
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
        else:  # UNBACKED
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
        if self._min is not None and self._max is not None and self._min > self._max:
            raise ValueError(
                f"min must be <= max, got min={self._min}, max={self._max}"
            )

    @classmethod
    def static(cls, name: str | None = None, *, value: int | None = None) -> "IntSpec":
        """Construct a STATIC `IntSpec`.

        ``value`` pins a concrete size; if ``None`` the value is taken from
        the example input at compile time.
        """
        return cls(name, type=IntSpecType.STATIC, value=value)

    @classmethod
    def backed(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        guarding_hint: int | None = None,
    ) -> "IntSpec":
        """Construct a BACKED `IntSpec`.

        ``guarding_hint`` is the concrete value the symbolic shape
        environment substitutes when a hint is needed for reasoning or
        codegen.
        """
        return cls(
            name,
            type=IntSpecType.BACKED,
            min=min,
            max=max,
            guarding_hint=guarding_hint,
        )

    @classmethod
    def unbacked(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> "IntSpec":
        """Construct an UNBACKED `IntSpec`.

        ``optimization_hint`` is used by downstream codegen (e.g. inductor
        autotuning) only; it never participates in symbolic reasoning.
        """
        return cls(
            name,
            type=IntSpecType.UNBACKED,
            min=min,
            max=max,
            optimization_hint=optimization_hint,
        )

    # -- fluent setters ----------------------------------------------------
    #
    # Each setter mutates in place, revalidates, and returns ``self`` for
    # chaining. Per-mode validity is enforced on each set, so e.g.
    # ``IntSpec.static("x").guarding_hint(10)`` raises ``ValueError``.

    def _try_set(self, slot: str, new_value: Any) -> None:
        # Atomic: if ``_validate`` rejects the new state, roll back so the
        # spec stays in a consistent state for the caller.
        old = getattr(self, slot)
        setattr(self, slot, new_value)
        try:
            self._validate()
        except Exception:
            setattr(self, slot, old)
            raise

    def name(self, value: str) -> "IntSpec":
        self._try_set("_name", value)
        return self

    def min(self, value: int) -> "IntSpec":
        self._try_set("_min", value)
        return self

    def max(self, value: int) -> "IntSpec":
        self._try_set("_max", value)
        return self

    def value(self, value: int) -> "IntSpec":
        self._try_set("_value", value)
        return self

    def guarding_hint(self, value: int) -> "IntSpec":
        self._try_set("_guarding_hint", value)
        return self

    def optimization_hint(self, value: int) -> "IntSpec":
        self._try_set("_optimization_hint", value)
        return self

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = []
        for slot in self.__slots__:
            val = getattr(self, slot)
            if slot == "_type":
                parts.append(f"type={val.name}")
            elif slot == "_name":
                if val is not None:
                    parts.append(f"name={val!r}")
            elif val is not None:
                parts.append(f"{slot[1:]}={val}")
        return f"IntSpec({', '.join(parts)})"


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``IntSpec | None`` with length equal to the
    tensor's dim. ``None`` entries inherit the default dynamism policy from
    the compile context.

    Construct from any of:

    - ``int`` — number of dims; all entries start as ``None``.
    - ``list`` / ``tuple`` of ``IntSpec | None`` — dim is inferred from
      length, entries used as-is.
    - ``dict[int, IntSpec | None]`` — sparse per-dim spec; dim is inferred
      from ``max(keys) + 1``. Empty dict rejected.

    Example::
        TensorSpec(3)  # rank 3, all None
        TensorSpec([IntSpec.backed("batch"), None])  # rank 2, dim 0 backed
        TensorSpec({0: IntSpec.backed("batch")})  # rank 1, dim 0 backed
    """

    def __init__(
        self,
        arg: int
        | list[IntSpec | None]
        | tuple[IntSpec | None, ...]
        | dict[int, IntSpec | None],
    ) -> None:
        if isinstance(arg, int):
            self._dim = arg
            self._specs: list[IntSpec | None] = [None] * arg
        elif isinstance(arg, (list, tuple)):
            self._dim = len(arg)
            self._specs = list(arg)
        elif isinstance(arg, dict):
            self._dim = max(arg.keys()) + 1
            self._specs = [None] * self._dim
            for k, v in arg.items():
                self._specs[k] = v
        else:
            raise TypeError(
                f"TensorSpec expects int / list / tuple / dict, "
                f"got {type(arg).__name__}"
            )

    def dim(self, index: int, spec: IntSpec) -> "TensorSpec":
        """Set the spec at ``index`` and return ``self`` for chaining."""
        self._specs[index] = spec
        return self

    def __getitem__(self, index: int) -> IntSpec | None:
        return self._specs[index]

    def __setitem__(self, index: int, spec: IntSpec | None) -> None:
        self._specs[index] = spec

    def __len__(self) -> int:
        return self._dim

    def __iter__(self) -> Iterator[IntSpec | None]:
        return iter(self._specs)

    def __repr__(self) -> str:
        entries = ", ".join(repr(spec) for spec in self._specs)
        return f"TensorSpec([{entries}])"

    # No ``__eq__`` / ``__hash__``: matches :class:`IntSpec`'s design — specs
    # are immutable compile-time inputs compared via ``repr()`` when needed.
    # Value-based equality would force cache keys to drift with object
    # identity and conflict with the AOT-snapshot invariant.


# ``LeafSpec`` is anything that can sit at a leaf of the spec tree (a
# single-int spec or a per-dim tensor spec). Used in ``ObjectSpec``'s
# field-value annotation, where the recursion is ``LeafSpec | ObjectSpec``.
LeafSpec = IntSpec | TensorSpec


class _KeyKind(enum.Enum):
    """How an ``ObjectSpec`` entry is keyed at lookup time.

    Determines which pytree ``KeyEntry`` is emitted by
    ``flatten_with_keys``, and therefore which ``Source`` shape in the
    dynamo builder the spec matches.
    """

    ITEM = "field"  # → ``MappingKey``; matches ``LocalSource`` / ``GetItemSource``
    ATTR = "attr"  # → ``GetAttrKey``; matches ``AttrSource``


class ObjectSpec:
    """Top-level dynamic-shape spec keyed by argument name.

    Each entry has a *key kind* — either item-access (dict-keyed, the
    default — produces ``MappingKey`` paths) or attribute-access
    (produces ``GetAttrKey`` paths). Use ``.field`` for item-keyed
    entries, ``.attr`` for attribute-keyed ones. Each kind matches a
    different ``Source`` shape in the dynamo builder:

    - ``.field("x", spec)`` → ``MappingKey("x")`` → matches
      ``LocalSource("x")`` and ``GetItemSource(..., "x")``.
    - ``.attr("weight", spec)`` → ``GetAttrKey("weight")`` → matches
      ``AttrSource(..., "weight")`` (e.g. ``nn.Module`` attribute
      access).

    Construct via a one-shot dict (all entries item-keyed) or
    incrementally via the fluent setters; mix freely::

        # Dict form (all ``.field`` / ``MappingKey``)
        ObjectSpec(
            {"x": TensorSpec([IntSpec.backed("batch")]), "n": IntSpec.backed("n")}
        )

        # Fluent form, mixing dict-key and attribute access
        (
            ObjectSpec()
            .field("x", TensorSpec([IntSpec.backed("batch")]))
            .attr(
                "model", ObjectSpec().attr("weight", TensorSpec([IntSpec.backed("h")]))
            )
        )

    Recursion is allowed: an ``ObjectSpec`` may hold nested
    ``ObjectSpec`` values to describe deeper trees. The initial
    integration only consumes top-level entries; nested entries are
    silently inert until the source-chain → keypath translation in the
    dynamo builder is extended.
    """

    def __init__(
        self,
        fields: dict[str, "LeafSpec | ObjectSpec"] | None = None,
    ) -> None:
        # Each entry is ``(kind, spec)``; insertion order preserved.
        self._fields: dict[str, tuple[_KeyKind, IntSpec | TensorSpec | ObjectSpec]] = {}
        if fields is not None:
            for k, v in fields.items():
                self._fields[k] = (_KeyKind.ITEM, v)

    def field(self, name: str, spec: "LeafSpec | ObjectSpec") -> "ObjectSpec":
        """Item-keyed entry (dict-like). Sets the spec at ``name`` and
        returns ``self`` for chaining. Path entry is ``MappingKey``."""
        self._fields[name] = (_KeyKind.ITEM, spec)
        return self

    def attr(self, name: str, spec: "LeafSpec | ObjectSpec") -> "ObjectSpec":
        """Attribute-keyed entry (object attribute access). Sets the spec
        at ``name`` and returns ``self`` for chaining. Path entry is
        ``GetAttrKey``."""
        self._fields[name] = (_KeyKind.ATTR, spec)
        return self

    def __getitem__(self, name: str) -> "LeafSpec | ObjectSpec":
        return self._fields[name][1]

    def __setitem__(self, name: str, spec: "LeafSpec | ObjectSpec") -> None:
        # Preserve existing key kind on overwrite; default to ITEM for
        # new entries (matches the dict-form constructor).
        kind = self._fields[name][0] if name in self._fields else _KeyKind.ITEM
        self._fields[name] = (kind, spec)

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def items(self) -> Any:
        """Yield ``(name, spec)`` pairs in insertion order — kind is an
        implementation detail of path translation."""
        return ((k, v) for k, (_, v) in self._fields.items())

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{kind.value}({name!r}): {spec!r}"
            for name, (kind, spec) in self._fields.items()
        )
        return f"ObjectSpec({{{entries}}})"

    @classmethod
    def match(cls, obj: Any) -> Any:
        """Auto-derive a default spec scaffold mirroring ``obj``'s structure.

        Mapping per input kind:

        - ``torch.Tensor``         → ``TensorSpec(obj.ndim)`` (all dims default-policy)
        - ``int`` (not ``bool``)   → ``IntSpec.static()``
        - ``dict``                 → native ``dict`` of ``match(v)`` per entry
        - ``list`` / ``tuple``     → native ``list`` / ``tuple`` of ``match(v)`` per entry
        - ``torch.nn.Module``      → ``ObjectSpec`` with ``.attr`` per child
                                     module, parameter, and buffer (one level —
                                     submodules recurse via the ``match`` call)
        - other                    → ``TypeError``

        ``bool`` is rejected explicitly (it inherits from ``int`` and almost
        always indicates a user mistake).
        """
        if isinstance(obj, torch.Tensor):
            return TensorSpec(obj.ndim)
        if isinstance(obj, bool):
            raise TypeError(f"ObjectSpec.match cannot derive a spec for bool {obj!r}")
        if isinstance(obj, int):
            return IntSpec.static()
        if isinstance(obj, dict):
            return {k: cls.match(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(cls.match(v) for v in obj)
        if isinstance(obj, torch.nn.Module):
            os = cls()
            # Walk one level: child modules, parameters, buffers — each child
            # module recurses via ``match``. ``recurse=False`` keeps the
            # walk shallow so the returned ObjectSpec mirrors the module
            # tree's hierarchy rather than a flat name table.
            for name, child in obj.named_children():
                os.attr(name, cls.match(child))
            for name, p in obj.named_parameters(recurse=False):
                os.attr(name, cls.match(p))
            for name, b in obj.named_buffers(recurse=False):
                os.attr(name, cls.match(b))
            return os
        raise TypeError(
            f"ObjectSpec.match cannot derive a spec for {type(obj).__name__}"
        )

    # No ``__eq__`` / ``__hash__``: same call as :class:`IntSpec` /
    # :class:`TensorSpec`.


# -- pytree registration -----------------------------------------------------
#
# ``TensorSpec`` is a list-like container of per-dim ``IntSpec | None`` —
# register it so ``tree_flatten_with_path`` exposes each per-dim entry as a
# leaf with a ``SequenceKey(i)`` path entry.
#
# ``IntSpec`` is intentionally *not* registered: pytree treats unregistered
# classes as opaque leaves, which is the correct behavior — flattening should
# stop at ``IntSpec``.


def _tensorspec_flatten(ts: TensorSpec) -> tuple[list[Any], None]:
    return list(ts), None


def _tensorspec_unflatten(children: Any, _context: Any) -> TensorSpec:
    return TensorSpec(list(children))


def _tensorspec_flatten_with_keys(
    ts: TensorSpec,
) -> tuple[list[tuple[Any, Any]], None]:
    return [(pytree.SequenceKey(i), spec) for i, spec in enumerate(ts)], None


pytree.register_pytree_node(
    TensorSpec,
    _tensorspec_flatten,
    _tensorspec_unflatten,
    flatten_with_keys_fn=_tensorspec_flatten_with_keys,
)


# ``ObjectSpec`` flattens to its field values; the per-entry kind
# determines which pytree ``KeyEntry`` ``flatten_with_keys`` emits —
# ``MappingKey`` for ``.field`` (item access), ``GetAttrKey`` for
# ``.attr`` (attribute access). The context preserved across
# flatten/unflatten is the list of ``(name, kind)`` pairs.


def _objectspec_flatten(
    os: ObjectSpec,
) -> tuple[list[Any], list[tuple[str, _KeyKind]]]:
    return (
        [v for _, v in os._fields.values()],
        [(name, kind) for name, (kind, _) in os._fields.items()],
    )


def _objectspec_unflatten(values: Any, context: Any) -> ObjectSpec:
    out = ObjectSpec()
    for (name, kind), v in zip(context, values):
        out._fields[name] = (kind, v)
    return out


def _objectspec_flatten_with_keys(
    os: ObjectSpec,
) -> tuple[list[tuple[Any, Any]], list[tuple[str, _KeyKind]]]:
    def _key(name: str, kind: _KeyKind) -> Any:
        if kind is _KeyKind.ATTR:
            return pytree.GetAttrKey(name)
        return pytree.MappingKey(name)

    return (
        [(_key(name, kind), spec) for name, (kind, spec) in os._fields.items()],
        [(name, kind) for name, (kind, _) in os._fields.items()],
    )


pytree.register_pytree_node(
    ObjectSpec,
    _objectspec_flatten,
    _objectspec_unflatten,
    flatten_with_keys_fn=_objectspec_flatten_with_keys,
)


# -- compile-time dispatch ---------------------------------------------------
#
# At ``torch.compile`` entry, the user's ``dynamic_shapes`` (any
# pytree-compatible structure with ``IntSpec`` / ``TensorSpec`` / ``None``
# leaves matching the function's argument tree) is flattened into a
# ``path -> IntSpec`` map and stashed in ``_active_dynamic_shapes``. The
# dynamo builder consults this map during input wrapping via
# :func:`get_active_spec_for_arg` (scalar args) and
# :func:`get_active_spec_for_dim` (tensor dims).


_active_dynamic_shapes: ContextVar[dict[tuple[Any, ...], IntSpec] | None] = ContextVar(
    "_active_dynamic_shapes", default=None
)


def _flatten_dynamic_shapes(shapes: Any) -> dict[tuple[Any, ...], IntSpec]:
    """Flatten the user's spec tree into a ``path -> IntSpec`` map.

    ``None`` leaves and any non-``IntSpec`` leaf are dropped — they
    represent "no override" and the consumer should fall through to the
    default policy.
    """
    leaves_with_paths, _ = pytree.tree_flatten_with_path(shapes)
    return {
        tuple(path): leaf
        for path, leaf in leaves_with_paths
        if isinstance(leaf, IntSpec)
    }


def get_active_spec_for_arg(arg_name: str) -> IntSpec | None:
    """Return the spec for a top-level scalar argument, or ``None``."""
    path_map = _active_dynamic_shapes.get()
    if path_map is None:
        return None
    return path_map.get((pytree.MappingKey(arg_name),))


def get_active_spec_for_dim(arg_name: str, dim: int) -> IntSpec | None:
    """Return the spec for a top-level tensor argument's specific dim, or
    ``None``."""
    path_map = _active_dynamic_shapes.get()
    if path_map is None:
        return None
    return path_map.get((pytree.MappingKey(arg_name), pytree.SequenceKey(dim)))
