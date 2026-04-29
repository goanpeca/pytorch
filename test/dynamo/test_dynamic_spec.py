# Owner(s): ["module: dynamo"]

import inspect

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.fx.experimental._config as _fx_experimental_config
import torch.utils._pytree as pytree
from torch._dynamo.decorators import mark_static, mark_unbacked, maybe_mark_dynamic
from torch._dynamo.dynamic_spec import (
    _active_dynamic_shapes,
    _flatten_dynamic_shapes,
    get_active_spec_for_arg,
    get_active_spec_for_dim,
    IntSpec,
    IntSpecType,
    ObjectSpec,
    TensorSpec,
)
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
)


def _tensor_placeholder_shape(gm):
    """Return the shape of the first tensor-typed placeholder in ``gm``."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ev = node.meta.get("example_value")
            if isinstance(ev, torch.Tensor):
                return ev.shape
    raise AssertionError("no tensor placeholder found")


# Applies per-dim IntSpec to a tensor through ``mark_*`` on each call.
# ``shape_spec`` must be a ``TensorSpec`` (or ``None`` to skip).
def _apply_intspec_to_tensor(tensor, shape_spec):
    if not isinstance(shape_spec, TensorSpec):
        return
    for idx, spec in enumerate(shape_spec):
        if spec is None:
            continue
        if spec._type is IntSpecType.STATIC:
            mark_static(tensor, idx)
        elif spec._type is IntSpecType.BACKED:
            maybe_mark_dynamic(tensor, idx)
        elif spec._type is IntSpecType.UNBACKED:
            mark_unbacked(tensor, idx)


def _compile_with_dynamic_shapes(fn, dynamic_shapes, **compile_kwargs):
    """Compile ``fn`` and apply ``dynamic_shapes`` specs on every call.

    On each call, the wrapper inspects ``dynamic_shapes`` and calls the
    appropriate ``mark_static`` / ``maybe_mark_dynamic`` / ``mark_unbacked``
    on each tensor argument before forwarding to the compiled function.
    """

    compiled = torch.compile(fn, **compile_kwargs)
    sig = inspect.signature(fn)

    @torch._dynamo.disable
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, shape_spec in dynamic_shapes.items():
            if name in bound.arguments:
                arg = bound.arguments[name]
                if isinstance(arg, torch.Tensor):
                    _apply_intspec_to_tensor(arg, shape_spec)
        return compiled(*bound.args, **bound.kwargs)

    return wrapper


class TestIntSpecConstruction(TestCase):
    """Construction via the classmethod factories.

    Field reads use the private slots (``s._name``, ``s._min``, etc.) since
    the fluent setters are write-only.
    """

    def test_static(self):
        s = IntSpec.static("x", value=10)
        self.assertEqual(s._name, "x")
        self.assertEqual(s._type, IntSpecType.STATIC)
        self.assertEqual(s._value, 10)

    def test_static_no_value(self):
        s = IntSpec.static()
        self.assertEqual(s._type, IntSpecType.STATIC)
        self.assertIsNone(s._value)

    def test_backed(self):
        s = IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        self.assertEqual(s._name, "batch")
        self.assertEqual(s._type, IntSpecType.BACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)
        self.assertEqual(s._guarding_hint, 32)

    def test_unbacked(self):
        s = IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        self.assertEqual(s._type, IntSpecType.UNBACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 2048)
        self.assertEqual(s._optimization_hint, 512)

    def test_type_required_on_init(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec("x")  # no type kwarg
        # Python-generated message; format pinned for 3.10+.
        self.assertEqual(
            str(cm.exception),
            "IntSpec.__init__() missing 1 required positional argument: 'type'",
        )

    def test_type_not_none(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec("x", type=None)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec type must be an IntSpecType, got None",
        )

    def test_type_as_positional_arg(self):
        s = IntSpec("batch", IntSpecType.BACKED, min=1, max=64)
        self.assertEqual(s._name, "batch")
        self.assertEqual(s._type, IntSpecType.BACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)

    def test_static_with_positional_int_rejected(self):
        # ``IntSpec.static(10)`` would silently bind 10 to ``name``. Must
        # fail with a clear redirect to the kwarg form.
        with self.assertRaises(TypeError) as cm:
            IntSpec.static(10)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )

    def test_backed_with_positional_int_rejected(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec.backed(5)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.backed(guarding_hint=5))",
        )

    def test_unbacked_with_positional_int_rejected(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec.unbacked(5)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.unbacked(optimization_hint=5))",
        )

    def test_name_wrong_type_on_init(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec(10, IntSpecType.STATIC)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )

    def test_repr(self):
        # Anonymous, no fields set.
        self.assertEqual(repr(IntSpec.static()), "IntSpec(type=STATIC)")
        # Named + value.
        self.assertEqual(
            repr(IntSpec.static("x", value=10)),
            "IntSpec(name='x', type=STATIC, value=10)",
        )
        # BACKED with full set of fields.
        self.assertEqual(
            repr(IntSpec.backed("batch", min=1, max=64, guarding_hint=32)),
            "IntSpec(name='batch', type=BACKED, min=1, max=64, guarding_hint=32)",
        )
        # UNBACKED with full set of fields.
        self.assertEqual(
            repr(IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)),
            "IntSpec(name='seq', type=UNBACKED, min=1, max=2048, optimization_hint=512)",
        )


class TestIntSpecTypeImmutable(TestCase):
    """Only ``type`` is pinned. All other fields can be reassigned via the
    fluent setters. This class verifies the type-pin guard, slot discipline
    (no new attrs, cannot delete), and that the mutable fields can in fact
    be updated."""

    def test_type_reassign_via_private_slot_rejected(self):
        # The backing ``_type`` slot is locked: our ``__setattr__``
        # guard catches reassignment. Reads through ``_type`` are fine.
        s = IntSpec.backed("x")
        with self.assertRaises(AttributeError) as cm:
            s._type = IntSpecType.STATIC
        self.assertEqual(
            str(cm.exception),
            "IntSpec type is immutable; cannot reassign",
        )
        self.assertIs(s._type, IntSpecType.BACKED)

    def test_no_fluent_type_reset(self):
        # IntSpec has no instance method that reassigns type. The mode-named
        # factories are classmethods: calling one "on an instance" returns a
        # fresh IntSpec and does not mutate the original.
        s = IntSpec.static("x")
        new = IntSpec.backed("x")
        self.assertIs(s._type, IntSpecType.STATIC)
        self.assertIs(new._type, IntSpecType.BACKED)
        self.assertIsNot(s, new)

    def test_cannot_add_new_attribute(self):
        # __slots__ rejects unknown attributes at the slot layer; message
        # is Python-built and version-dependent, so we only assert the type.
        s = IntSpec.static("x")
        with self.assertRaises(AttributeError):
            s.brand_new_field = 1  # type: ignore[attr-defined]

    def test_cannot_assign_to_method_name(self):
        # ``value`` / ``min`` / ``max`` / ``guarding_hint`` /
        # ``optimization_hint`` are method names backed by ``_value`` /
        # ``_min`` / etc. slots. Direct attribute assignment under those
        # public names has no matching slot and is rejected.
        s = IntSpec.static("x", value=10)
        for attr in ("value", "min", "max", "guarding_hint", "optimization_hint"):
            with self.assertRaises(AttributeError):
                setattr(s, attr, 20)

    def test_cannot_delete_attribute(self):
        s = IntSpec.backed("x", guarding_hint=32)
        with self.assertRaises(AttributeError) as cm:
            del s._guarding_hint
        self.assertEqual(
            str(cm.exception),
            "IntSpec attribute '_guarding_hint' cannot be deleted",
        )


class TestIntSpecFluent(TestCase):
    """Fluent setters (``name`` / ``min`` / ``max`` / ``value`` /
    ``guarding_hint`` / ``optimization_hint``) mutate the receiver in place
    and return ``self`` for chaining. Each chain is equivalent to the
    kwargs-only factory form."""

    def test_setter_returns_self_for_chaining(self):
        base = IntSpec.unbacked("seq")
        chained = base.min(1).max(2048)
        # Mutated in place, same object back.
        self.assertIs(base, chained)
        self.assertEqual(base._min, 1)
        self.assertEqual(base._max, 2048)

    def test_fluent_preserves_existing_fields(self):
        s = IntSpec.backed("batch", min=1, guarding_hint=32).max(64)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)
        self.assertEqual(s._guarding_hint, 32)

    def test_failed_setter_rolls_back_per_mode(self):
        # Per-mode rejection (guarding_hint on STATIC) must roll the slot
        # back so the spec stays in a consistent state.
        s = IntSpec.static("x", value=10)
        with self.assertRaises(ValueError):
            s.guarding_hint(99)
        self.assertIsNone(s._guarding_hint)
        self.assertEqual(s._value, 10)

    def test_failed_setter_rolls_back_min_max(self):
        # Cross-field check (min <= max) also rolls back.
        s = IntSpec.backed("x", min=10, max=20)
        with self.assertRaises(ValueError):
            s.max(5)
        self.assertEqual(s._max, 20)


@instantiate_parametrized_tests
class TestIntSpecRejectionRules(TestCase):
    """Per-mode field-rejection rules, exercised via both entry points.

    Each rule fires inside :meth:`IntSpec._validate`, reached from two
    user-visible paths:

    - ``init``: direct kwargs at the raw constructor, e.g.
      ``IntSpec("x", IntSpecType.STATIC, guarding_hint=10)``.
    - ``fluent``: fluent setter chained off a factory, e.g.
      ``IntSpec.static("x").guarding_hint(10)``.

    Both paths produce the same error message because the constructor
    and the fluent setter both call ``_validate`` (the fluent setter via
    ``_try_set``). Parametrizing the entry-point axis keeps a single
    source of truth per rule.
    """

    @parametrize("entry", ["init", "fluent"])
    @parametrize(
        "IntSpecFactory,mode",
        [
            (IntSpec.backed, IntSpecType.BACKED),
            (IntSpec.unbacked, IntSpecType.UNBACKED),
        ],
    )
    def test_value_rejected_on_non_static(self, IntSpecFactory, mode, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, value=42)  # noqa: E731
        else:
            ctor = lambda: IntSpecFactory("x").value(42)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(str(cm.exception), "value is only valid for STATIC IntSpec")

    @parametrize("entry", ["init", "fluent"])
    @parametrize(
        "IntSpecFactory,mode",
        [
            (IntSpec.static, IntSpecType.STATIC),
            (IntSpec.unbacked, IntSpecType.UNBACKED),
        ],
    )
    def test_guarding_hint_rejected_on_non_backed(self, IntSpecFactory, mode, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, guarding_hint=10)  # noqa: E731
        else:
            ctor = lambda: IntSpecFactory("x").guarding_hint(10)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception), "guarding_hint is only valid for BACKED IntSpec"
        )

    @parametrize("entry", ["init", "fluent"])
    @parametrize(
        "IntSpecFactory,mode",
        [
            (IntSpec.static, IntSpecType.STATIC),
            (IntSpec.backed, IntSpecType.BACKED),
        ],
    )
    def test_optimization_hint_rejected_on_non_unbacked(
        self, IntSpecFactory, mode, entry
    ):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, optimization_hint=10)  # noqa: E731
        else:
            ctor = lambda: IntSpecFactory("x").optimization_hint(10)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            "optimization_hint is only valid for UNBACKED IntSpec",
        )

    @parametrize("entry", ["init", "fluent"])
    @parametrize("field", ["min", "max"])
    def test_min_max_rejected_on_static(self, field, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", IntSpecType.STATIC, **{field: 1})  # noqa: E731
        else:
            ctor = lambda: getattr(IntSpec.static("x"), field)(1)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC",
        )

    @parametrize("IntSpecFactory", [IntSpec.backed, IntSpec.unbacked])
    def test_IntSpec_min_greater_than_max_rejected(self, IntSpecFactory):
        with self.assertRaises(ValueError) as cm:
            IntSpecFactory("x", min=100, max=1)
        self.assertEqual(
            str(cm.exception),
            "min must be <= max, got min=100, max=1",
        )

    # Each case: (field, bad_value, factory_name, expected_type_name).
    # ``factory_name`` is the factory whose mode allows the field — we
    # exercise the type check on the valid mode since mode-rejection is
    # already covered by the four tests above.
    _TYPE_CASES = [
        ("value", "10", "static", "str"),
        ("min", 1.5, "backed", "float"),
        ("max", "64", "backed", "str"),
        ("guarding_hint", True, "backed", "bool"),
        ("optimization_hint", 1.0, "unbacked", "float"),
    ]

    @parametrize("entry", ["init", "fluent"])
    @parametrize("field,bad,factory_name,type_name", _TYPE_CASES)
    def test_field_type_rejected(self, field, bad, factory_name, type_name, entry):
        factory = getattr(IntSpec, factory_name)
        if entry == "init":
            ctor = lambda: factory("x", **{field: bad})  # noqa: E731
        else:
            ctor = lambda: getattr(factory("x"), field)(bad)  # noqa: E731
        with self.assertRaises(TypeError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            f"IntSpec.{field} must be int or None, got {type_name}",
        )


class TestTensorSpecConstruction(TestCase):
    """Construction and list-like interface."""

    def test_basic(self):
        ts = TensorSpec(3)
        self.assertEqual(ts._dim, 3)
        for spec in ts:
            self.assertIsNone(spec)

    def test_zero_dim(self):
        ts = TensorSpec(0)
        self.assertEqual(ts._dim, 0)

    def test_list_construction(self):
        static_spec = IntSpec.static(value=10)
        backed_spec = IntSpec.backed(min=1)
        ts = TensorSpec([static_spec, None, backed_spec])
        self.assertEqual(
            repr(ts),
            f"TensorSpec([{repr(static_spec)}, None, {repr(backed_spec)}])",
        )

    def test_dict_construction(self):
        backed_spec = IntSpec.backed("h")
        static_spec = IntSpec.static()
        ts = TensorSpec({0: backed_spec, 2: static_spec})
        self.assertEqual(
            repr(ts), f"TensorSpec([{repr(backed_spec)}, None, {repr(static_spec)}])"
        )

    def test_unsupported_input_type_rejected(self):
        with self.assertRaisesRegex(TypeError, "expects int / list / tuple / dict"):
            TensorSpec("not a spec")  # type: ignore[arg-type]

    def test_getitem_setitem(self):
        ts = TensorSpec(2)
        spec = IntSpec.backed("batch", min=1)
        ts[0] = spec
        self.assertIs(ts[0], spec)
        self.assertIsNone(ts[1])

    def test_iter(self):
        ts = TensorSpec(2)
        spec = IntSpec.static(value=5)
        ts[0] = spec
        items = list(ts)
        self.assertEqual(len(items), 2)
        self.assertIs(items[0], spec)
        self.assertIsNone(items[1])

    def test_index_out_of_range(self):
        ts = TensorSpec(2)
        with self.assertRaises(IndexError):
            ts[5]

    def test_sparse_set(self):
        ts = TensorSpec(4)
        ts.dim(1, IntSpec.backed("h"))
        ts.dim(3, IntSpec.backed("w"))
        self.assertIsNone(ts[0])
        self.assertIsNotNone(ts[1])
        self.assertIsNone(ts[2])
        self.assertIsNotNone(ts[3])


@skipIfTorchDynamo()
class TestTensorSpecCompile(TestCase):
    """TensorSpec + compile integration via the test-local
    `_compile_with_dynamic_shapes` helper. Same scaffolding path as
    `TestIntSpecCompile` — the helper walks per-dim ``IntSpec`` from
    the TensorSpec and applies ``mark_*`` to the tensor.
    """

    def test_tensorspec_list_init_recompile_progression(self):
        """TensorSpec built from a list: dim 0 BACKED absorbs shape changes;
        dim 1 STATIC forces recompile per distinct value.

        Final graph has a backed SymInt at dim 0 and a concrete int at dim 1.
        """
        backend = EagerAndRecordGraphs()
        ts = TensorSpec([IntSpec.backed("batch"), IntSpec.static()])
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": ts},
            backend=backend,
        )

        fn(torch.randn(4, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary dim 0 (BACKED absorbs it → no new compile).
        fn(torch.randn(8, 3))
        fn(torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary dim 1 (STATIC pins it → each distinct value recompiles).
        fn(torch.randn(16, 5))
        self.assertEqual(len(backend.graphs), 2)
        fn(torch.randn(16, 7))
        self.assertEqual(len(backend.graphs), 3)

        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)
        self.assertIsInstance(shape[1], int)
        self.assertEqual(shape[1], 7)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_tensorspec_none_entry_inherits_automatic_dynamic(self):
        """A ``None`` entry doesn't mark the dim — it falls through to the
        existing default policy: dynamo's automatic-dynamic
        (``torch._dynamo.config.automatic_dynamic_shapes``, default True).
        That means specialize on first call, promote to backed on first
        variation.

        The decorator pins the config to ``True`` so the test is robust to
        any CI environment that might flip the default. This mainly test
        config inheritance, not the default policy itself.
        """
        backend = EagerAndRecordGraphs()
        ts = TensorSpec([IntSpec.backed("batch"), None])
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": ts},
            backend=backend,
        )

        fn(torch.randn(4, 3))  # specialize dim 1=3; BACKED dim 0
        self.assertEqual(len(backend.graphs), 1)

        fn(torch.randn(8, 3))  # dim 0 absorbed; dim 1 same → cache hit
        self.assertEqual(len(backend.graphs), 1)

        fn(torch.randn(8, 5))  # dim 1 promotes to backed → recompile
        self.assertEqual(len(backend.graphs), 2)

        fn(torch.randn(16, 7))  # dim 0 + dim 1 both backed → cache hit
        self.assertEqual(len(backend.graphs), 2)


@skipIfTorchDynamo()
class TestIntSpecCompile(TestCase):
    """Covers IntSpec + torch.compile integration using the local
    `_compile_with_dynamic_shapes` helper.

    Includes tests for captured-graph shape properties and for precedence
    rules between IntSpec annotations and compile-time dynamic settings.
    """

    def test_static_graph_has_concrete_shape(self):
        """STATIC dim appears as a concrete int in the captured graph; each
        distinct shape yields a new graph."""
        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": TensorSpec({0: IntSpec.static()})},
            backend=backend,
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        fn(torch.randn(12, 3))
        fn(torch.randn(4, 3))  # cache hit

        # STATIC bakes the concrete size into the trace: 3 distinct sizes
        # (4, 8, 12) → 3 graphs; the repeat size=4 reuses graph #1.
        self.assertEqual(len(backend.graphs), 3)
        for gm in backend.graphs:
            shape = _tensor_placeholder_shape(gm)
            self.assertIsInstance(shape[0], int)

    def test_backed_graph_has_backed_symbol(self):
        """BACKED dim appears as a backed SymInt in the final graph."""

        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0),
            {"x": TensorSpec({0: IntSpec.backed("batch")})},
            backend=backend,
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))

        # Pre-trace maybe_mark_dynamic → backed from call 1, no branches,
        # no 0/1 specialization → 1 compile covers all 5 sizes.
        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        # backed symbol: no free unbacked symbols
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)

    def test_unbacked_graph_has_unbacked_symbol(self):
        """UNBACKED dim appears as an unbacked SymInt; single compile covers all shapes."""
        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0),
            {"x": TensorSpec({0: IntSpec.unbacked("batch")})},
            backend=backend,
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))

        # UNBACKED dim produces a symbol with no backing value; no guards
        # can attach (no branch on size), and 0/1 specialization doesn't
        # apply to unbacked → 1 compile covers all 4 sizes.
        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertGreater(len(free_unbacked_symbols(shape[0])), 0)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        """A function that branches on size(0) must raise a data-dependent
        error when that dim is marked UNBACKED.

        The ``no_data_dependent_graph_break`` config flag disables dynamo's
        default rewrap of ``GuardOnDataDependentSymNode`` into ``UserError``,
        so the raw exception surfaces and we can assert on ``.cond`` directly.
        """

        def fn(x):
            if x.size(0) > 5:
                return x + 1
            return x - 1

        compiled = _compile_with_dynamic_shapes(
            fn,
            {"x": TensorSpec({0: IntSpec.unbacked()})},
            backend="eager",
            fullgraph=True,
        )
        with self.assertRaises(GuardOnDataDependentSymNode) as cm:
            compiled(torch.randn(10, 3))
        # The guard expression must reference a free unbacked symbol —
        # that's what makes it "data-dependent" and confirms our UNBACKED
        # spec actually produced an unbacked dim (backed would be ``s0``,
        # never raise DDE).
        free_syms = cm.exception.cond.free_symbols
        self.assertEqual(len(free_syms), 1)
        # ShapeEnv names unbacked symbols with a ``u`` prefix.
        (sym,) = free_syms
        self.assertTrue(
            str(sym).startswith("u"),
            msg=f"expected unbacked symbol (u-prefix), got {sym!r}",
        )

    def test_backed_branching_bounded_recompiles(self):
        """BACKED + branching on size(0) > 8 should produce exactly 2 compiles
        for inputs [4, 8, 16, 32, 64].

        Call sequence:
        - 4, 8  -> branch False, same compile
        - 16    -> branch True, second compile
        - 32, 64 -> branch True, cache hits

        Total compiles: 2 (one per branch path), unlike STATIC which compiles per shape.
        """
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0 if x.size()[0] > 8 else 1),
            {"x": TensorSpec({0: IntSpec.backed("batch")})},
            backend=cnt,
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        # Dim is backed from call 1; the ``> 8`` branch splits compiles by
        # path — 2 paths (True/False) → 2 compiles, regardless of shape count.
        self.assertEqual(cnt.frame_count, 2)

    def test_backed_zero_one_specialization(self):
        """BACKED symbols are specialized at 0 and 1 unconditionally.

        For ``[0, 1, 2, 4, 8]``:
        - n=0: specialized → Compile #1
        - n=1: specialized → Compile #2
        - n=2: backed SymInt with guard ``size >= 2`` → Compile #3
        - n=4, 8: cache hits on #3

        Total = 3.
        """

        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": TensorSpec({0: IntSpec.backed("batch")})},
            backend=cnt,
        )
        for n in [0, 1, 2, 4, 8]:
            fn(torch.randn(n, 3))
        # 0 and 1 each force their own specialized compile (PyTorch-wide
        # 0/1 rule, applies even to backed); sizes ≥ 2 share one backed
        # compile with guard ``size >= 2`` → 2 + 1 = 3 compiles.
        self.assertEqual(cnt.frame_count, 3)

    def test_backed_equality_branching(self):
        """BACKED + Python branch on ``==``: point specialization.

        For ``x.size(0) == 3`` against sizes ``[3, 4, 5, 6]``:
        - n=3: branch True, guard ``size == 3`` → Compile #1
        - n=4: guard fails → recompile, branch False, guard ``size != 3`` → Compile #2
        - n=5, 6: cache hits on #2

        Total = 2.
        """

        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1 if x.size()[0] == 3 else x - 1,
            {"x": TensorSpec({0: IntSpec.backed("batch")})},
            backend=cnt,
        )
        for n in [3, 4, 5, 6]:
            fn(torch.randn(n, 3))
        # ``== 3`` splits into two branch paths: point-specialization at
        # size=3 (compile #1) and the ``size != 3`` backed compile
        # (compile #2). Same count as the ``>`` case, different guards.
        self.assertEqual(cnt.frame_count, 2)

    def test_static_precedence_over_dynamic_true(self):
        """IntSpec.static() must win over compile(dynamic=True)."""
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": TensorSpec({0: IntSpec.static()})},
            backend=cnt,
            dynamic=True,
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        # Spec STATIC beats ``compile(dynamic=True)``: 2 distinct shapes
        # each get a specialized compile → 2 compiles. If ``dynamic=True``
        # had won, we'd see 1 compile with a backed SymInt.
        self.assertEqual(cnt.frame_count, 2)

    def test_backed_precedence_over_dynamic_false(self):
        """IntSpec.backed() must win over compile(dynamic=False).

        With the compile-context integration, the spec selects
        DimDynamic.DYNAMIC directly — the first call is already backed, no
        initial specialization, so a single compile covers all shapes.
        """
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0),
            {"x": TensorSpec({0: IntSpec.backed("batch")})},
            backend=cnt,
            dynamic=False,
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        # Spec BACKED beats ``compile(dynamic=False)``: dim is backed from
        # call 1 via the pre-trace mark, no branches, no 0/1 sizes → 1
        # compile. If ``dynamic=False`` had won, we'd see 5 (one per shape).
        self.assertEqual(cnt.frame_count, 1)


class TestObjectSpec(TestCase):
    """Construction, dict-like access, and recursion."""

    def test_empty(self):
        os = ObjectSpec()
        self.assertEqual(len(os), 0)

    def test_dict_construction(self):
        spec = IntSpec.backed("batch")
        os = ObjectSpec({"x": spec, "n": IntSpec.static()})
        self.assertEqual(len(os), 2)
        self.assertIn("x", os)
        self.assertIs(os["x"], spec)

    def test_fluent_field(self):
        spec_x = TensorSpec([IntSpec.backed("batch")])
        spec_n = IntSpec.unbacked("n")
        os = ObjectSpec().field("x", spec_x).field("n", spec_n)
        self.assertEqual(len(os), 2)
        self.assertIs(os["x"], spec_x)
        self.assertIs(os["n"], spec_n)

    def test_setitem(self):
        os = ObjectSpec()
        spec = IntSpec.backed("batch")
        os["x"] = spec
        self.assertIs(os["x"], spec)

    def test_iter_and_items(self):
        spec_x = IntSpec.backed("batch")
        spec_n = IntSpec.static()
        os = ObjectSpec({"x": spec_x, "n": spec_n})
        self.assertEqual(list(os), ["x", "n"])
        self.assertEqual(list(os.items()), [("x", spec_x), ("n", spec_n)])

    def test_repr(self):
        # Single field-keyed entry.
        os = ObjectSpec({"x": IntSpec.static()})
        self.assertEqual(
            repr(os),
            "ObjectSpec({field('x'): IntSpec(type=STATIC)})",
        )

    def test_repr_multiple_entries(self):
        # Multiple entries preserve insertion order.
        os = ObjectSpec({"x": IntSpec.static(), "n": IntSpec.backed("n")})
        self.assertEqual(
            repr(os),
            "ObjectSpec({"
            "field('x'): IntSpec(type=STATIC), "
            "field('n'): IntSpec(name='n', type=BACKED)"
            "})",
        )

    def test_repr_mixed_field_and_attr(self):
        os = (
            ObjectSpec()
            .field("x", IntSpec.backed("batch"))
            .attr("weight", TensorSpec([IntSpec.backed("h")]))
        )
        self.assertEqual(
            repr(os),
            "ObjectSpec({"
            "field('x'): IntSpec(name='batch', type=BACKED), "
            "attr('weight'): TensorSpec([IntSpec(name='h', type=BACKED)])"
            "})",
        )

    def test_repr_nested_objectspec(self):
        inner = ObjectSpec().attr("weight", TensorSpec([IntSpec.backed("h")]))
        outer = ObjectSpec().attr("model", inner).field("x", IntSpec.static())
        self.assertEqual(
            repr(outer),
            "ObjectSpec({"
            "attr('model'): ObjectSpec({"
            "attr('weight'): TensorSpec([IntSpec(name='h', type=BACKED)])}), "
            "field('x'): IntSpec(type=STATIC)"
            "})",
        )

    def test_attr_setter(self):
        spec = TensorSpec([IntSpec.backed("h")])
        os = ObjectSpec().attr("weight", spec)
        self.assertIs(os["weight"], spec)
        self.assertEqual(
            repr(os),
            "ObjectSpec({attr('weight'): "
            "TensorSpec([IntSpec(name='h', type=BACKED)])})",
        )

    def test_mixed_field_and_attr(self):
        # Interleaved ``.field`` (item-keyed) and ``.attr``
        # (attribute-keyed) entries preserve insertion order.
        spec_x = IntSpec.backed("batch")
        spec_w = TensorSpec([IntSpec.backed("h")])
        os = ObjectSpec().field("x", spec_x).attr("weight", spec_w)
        self.assertIs(os["x"], spec_x)
        self.assertIs(os["weight"], spec_w)
        self.assertEqual(list(os), ["x", "weight"])

    def test_recursive_nesting(self):
        # Recursion in the data class — nested ``ObjectSpec`` is accepted.
        # Initial integration only consumes top-level entries; nested
        # entries remain inert at compile time but the class itself
        # carries them through.
        inner = ObjectSpec({"weight": TensorSpec([IntSpec.backed("h")])})
        outer = ObjectSpec({"model": inner, "n": IntSpec.backed("n")})
        self.assertIs(outer["model"], inner)


class TestObjectSpecPytree(TestCase):
    """``tree_flatten_with_path`` produces the expected paths for
    ``ObjectSpec`` containing ``IntSpec`` / ``TensorSpec`` leaves."""

    def test_top_level_intspec_leaf(self):
        spec = IntSpec.backed("n")
        os = ObjectSpec({"n": spec})
        leaves_with_paths, _ = pytree.tree_flatten_with_path(os)
        self.assertEqual(len(leaves_with_paths), 1)
        path, leaf = leaves_with_paths[0]
        self.assertEqual(tuple(path), (pytree.MappingKey("n"),))
        self.assertIs(leaf, spec)

    def test_tensorspec_dims_get_sequence_keys(self):
        s0 = IntSpec.backed("batch")
        s1 = IntSpec.static()
        os = ObjectSpec({"x": TensorSpec([s0, s1])})
        leaves_with_paths, _ = pytree.tree_flatten_with_path(os)
        path_to_leaf = {tuple(p): leaf for p, leaf in leaves_with_paths}
        self.assertIs(path_to_leaf[(pytree.MappingKey("x"), pytree.SequenceKey(0))], s0)
        self.assertIs(path_to_leaf[(pytree.MappingKey("x"), pytree.SequenceKey(1))], s1)

    def test_flatten_dynamic_shapes_drops_none_leaves(self):
        # ``None`` per-dim entries flatten as ``None`` leaves and the
        # helper drops them — the consumer falls through to the default
        # policy at those slots.
        os = ObjectSpec({"x": TensorSpec([IntSpec.backed("batch"), None])})
        path_map = _flatten_dynamic_shapes(os)
        self.assertEqual(len(path_map), 1)
        self.assertIn((pytree.MappingKey("x"), pytree.SequenceKey(0)), path_map)

    def test_attr_emits_get_attr_key(self):
        # ``.attr`` entries get ``GetAttrKey`` paths instead of
        # ``MappingKey`` — matches ``AttrSource`` in the dynamo builder.
        spec = IntSpec.backed("h")
        os = ObjectSpec().attr("weight", spec)
        leaves_with_paths, _ = pytree.tree_flatten_with_path(os)
        self.assertEqual(len(leaves_with_paths), 1)
        path, leaf = leaves_with_paths[0]
        self.assertEqual(tuple(path), (pytree.GetAttrKey("weight"),))
        self.assertIs(leaf, spec)

    def test_mixed_kinds_emit_distinct_path_types(self):
        # In the same ObjectSpec, ``.field`` and ``.attr`` entries emit
        # different KeyEntry types on their respective paths.
        s_x = IntSpec.backed("batch")
        s_w = IntSpec.backed("h")
        os = ObjectSpec().field("x", s_x).attr("weight", s_w)
        leaves_with_paths, _ = pytree.tree_flatten_with_path(os)
        path_to_leaf = {tuple(p): leaf for p, leaf in leaves_with_paths}
        self.assertIs(path_to_leaf[(pytree.MappingKey("x"),)], s_x)
        self.assertIs(path_to_leaf[(pytree.GetAttrKey("weight"),)], s_w)


class TestObjectSpecContextVar(TestCase):
    """``_active_dynamic_shapes`` ContextVar + the lookup helpers."""

    def test_unset_context_returns_none(self):
        # Default state: helpers return None when no spec is active.
        self.assertIsNone(get_active_spec_for_arg("x"))
        self.assertIsNone(get_active_spec_for_dim("x", 0))

    def test_get_active_spec_for_arg(self):
        spec = IntSpec.backed("n")
        os = ObjectSpec({"n": spec})
        token = _active_dynamic_shapes.set(_flatten_dynamic_shapes(os))
        try:
            self.assertIs(get_active_spec_for_arg("n"), spec)
            self.assertIsNone(get_active_spec_for_arg("missing"))
        finally:
            _active_dynamic_shapes.reset(token)

    def test_get_active_spec_for_dim(self):
        s0 = IntSpec.backed("batch")
        s1 = IntSpec.static()
        os = ObjectSpec({"x": TensorSpec([s0, s1])})
        token = _active_dynamic_shapes.set(_flatten_dynamic_shapes(os))
        try:
            self.assertIs(get_active_spec_for_dim("x", 0), s0)
            self.assertIs(get_active_spec_for_dim("x", 1), s1)
            self.assertIsNone(get_active_spec_for_dim("x", 5))
            self.assertIsNone(get_active_spec_for_dim("missing", 0))
        finally:
            _active_dynamic_shapes.reset(token)


class TestObjectSpecMatch(TestCase):
    """``ObjectSpec.match(obj)`` auto-derives a default spec scaffold
    matching ``obj``'s structure."""

    def test_match_tensor(self):
        x = torch.randn(2, 3, 4)
        spec = ObjectSpec.match(x)
        self.assertIsInstance(spec, TensorSpec)
        self.assertEqual(spec._dim, 3)
        # All dims default-policy.
        for i in range(3):
            self.assertIsNone(spec[i])

    def test_match_int(self):
        spec = ObjectSpec.match(5)
        self.assertIsInstance(spec, IntSpec)
        self.assertIs(spec._type, IntSpecType.STATIC)

    def test_match_bool_rejected(self):
        with self.assertRaisesRegex(TypeError, "bool"):
            ObjectSpec.match(True)

    def test_match_dict(self):
        result = ObjectSpec.match({"x": torch.randn(2, 3), "n": 4})
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["x"], TensorSpec)
        self.assertEqual(result["x"]._dim, 2)
        self.assertIsInstance(result["n"], IntSpec)

    def test_match_list_and_tuple_preserve_type(self):
        as_list = ObjectSpec.match([torch.randn(2), torch.randn(3, 4)])
        self.assertIsInstance(as_list, list)
        self.assertEqual(as_list[0]._dim, 1)
        self.assertEqual(as_list[1]._dim, 2)

        as_tuple = ObjectSpec.match((torch.randn(2),))
        self.assertIsInstance(as_tuple, tuple)
        self.assertEqual(as_tuple[0]._dim, 1)

    def test_match_nn_module(self):
        # Single-level module: parameters become attr-keyed TensorSpec
        # entries; the result mirrors the module's attribute layout.
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Linear())
        self.assertIsInstance(spec, ObjectSpec)
        self.assertIn("weight", spec)
        self.assertIn("bias", spec)
        self.assertEqual(spec["weight"]._dim, 2)
        self.assertEqual(spec["bias"]._dim, 1)

    def test_match_nested_nn_module(self):
        # Nested module: child modules nest as ObjectSpec under .attr;
        # parameters of the outer module appear alongside.
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Outer())
        self.assertIsInstance(spec["inner"], ObjectSpec)
        self.assertIsInstance(spec["inner"]["weight"], TensorSpec)
        self.assertIsInstance(spec["bias"], TensorSpec)

    def test_match_nn_module_with_buffer(self):
        class WithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(2))
                self.register_buffer("running_mean", torch.zeros(2))

        spec = ObjectSpec.match(WithBuffer())
        self.assertIn("weight", spec)
        self.assertIn("running_mean", spec)
        self.assertEqual(spec["running_mean"]._dim, 1)

    def test_match_unknown_type_rejected(self):
        with self.assertRaisesRegex(TypeError, "cannot derive a spec for"):
            ObjectSpec.match("not a spec source")

    def test_match_tensor_repr(self):
        spec = ObjectSpec.match(torch.randn(2, 3))
        self.assertEqual(repr(spec), "TensorSpec([None, None])")

    def test_match_int_repr(self):
        self.assertEqual(repr(ObjectSpec.match(5)), "IntSpec(type=STATIC)")

    def test_match_dict_repr(self):
        # Native dict carries through; leaf reprs come from IntSpec /
        # TensorSpec themselves.
        result = ObjectSpec.match({"x": torch.randn(2), "n": 4})
        self.assertEqual(
            repr(result),
            "{'x': TensorSpec([None]), 'n': IntSpec(type=STATIC)}",
        )

    def test_match_list_repr(self):
        result = ObjectSpec.match([torch.randn(2), torch.randn(3, 4)])
        self.assertEqual(
            repr(result),
            "[TensorSpec([None]), TensorSpec([None, None])]",
        )

    def test_match_nn_module_repr(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Linear())
        self.assertEqual(
            repr(spec),
            "ObjectSpec({"
            "attr('weight'): TensorSpec([None, None]), "
            "attr('bias'): TensorSpec([None])"
            "})",
        )

    def test_match_nested_nn_module_repr(self):
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Outer())
        self.assertEqual(
            repr(spec),
            "ObjectSpec({"
            "attr('inner'): ObjectSpec({"
            "attr('weight'): TensorSpec([None, None])}), "
            "attr('bias'): TensorSpec([None])"
            "})",
        )


if __name__ == "__main__":
    run_tests()
