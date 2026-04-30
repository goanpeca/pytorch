import itertools
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from os import linesep
from typing import Any

import sympy

import torch
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import DefaultHandler, WrapperHandler
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import DelayReplaceLine, IndentedBuffer, OrderedSet
from torch._inductor.virtualized import OpsValue

from ...virtualized import V


_ACCUMULATOR_ARG_NAME = "accum"


class _SiLUReconstituter:
    """Detects the decomposed SiLU pattern and replaces it with ops.silu().

    Inductor decomposes silu(x) as x / (1 + exp(-x)), which produces ops:
      neg, exp, constant(1), add, truediv
    CUTLASS EVT has native silu support but cannot handle scalar constants
    from the decomposed form. This class wraps the ops handler to detect
    the pattern and emit a single ops.silu() call instead.

    Handles two patterns:
    - Pure SiLU: silu(load(buf))
    - SiLU * aux: silu(load(buf_a)) * load(buf_b)
    """

    def __init__(self, inner_fn: Any, index_vars: Any):
        self._inner_fn = inner_fn
        self._index_vars = index_vars
        self._pattern: str | None = None  # "silu" or "silu_mul"
        self._silu_input: _LoadCapture | None = None
        self._mul_input: _LoadCapture | None = None

    def detect(self) -> bool:
        """Run inner_fn through a recording handler to detect SiLU pattern."""
        from unittest.mock import patch

        from torch._inductor.ir import FlexibleLayout

        recorder = _OpRecorder()
        try:
            with (
                virtualized.V.set_ops_handler(recorder),  # type: ignore[arg-type]
                patch.object(FlexibleLayout, "allow_indexing", True),
            ):
                self._inner_fn(self._index_vars)
        except Exception:
            return False

        # Try pure SiLU first, then silu*mul
        result = recorder.detect_silu()
        if result is not None:
            self._pattern = "silu"
            self._silu_input = result
            return True

        result_mul = recorder.detect_silu_mul()
        if result_mul is not None:
            self._pattern = "silu_mul"
            self._silu_input, self._mul_input = result_mul
            return True

        return False

    def make_replacement_inner_fn(self) -> Any:
        """Create a replacement inner_fn using native silu()."""
        assert self._pattern is not None
        silu_input = self._silu_input

        if self._pattern == "silu":

            def silu_inner_fn(index: Any) -> Any:
                x = silu_input.load_fn(index)  # type: ignore[union-attr]
                return virtualized.ops.silu(x)  # type: ignore[attr-defined]

            return silu_inner_fn

        else:  # silu_mul
            mul_input = self._mul_input

            def silu_mul_inner_fn(index: Any) -> Any:
                x = silu_input.load_fn(index)  # type: ignore[union-attr]
                y = mul_input.load_fn(index)  # type: ignore[union-attr]
                return virtualized.ops.mul(virtualized.ops.silu(x), y)  # type: ignore[attr-defined]

            return silu_mul_inner_fn


class _LoadCapture:
    """Captures a load operation for replay in the replacement inner_fn."""

    def __init__(self, name: str):
        self.name = name

    def load_fn(self, index: Any) -> Any:
        graph = virtualized.V.graph
        buf = graph.name_to_buffer.get(self.name) or graph.graph_inputs.get(self.name)
        if buf is None:
            raise KeyError(f"Buffer {self.name!r} not found in graph buffers or inputs")
        return buf.make_loader()(index)


@contextmanager
def reconstitute_silu(node: ComputedBuffer) -> Generator[None, None, None]:
    """Context manager that temporarily replaces decomposed SiLU with native silu().

    Use this around any call to node.get_store_function()() where the node's
    inner_fn may contain the decomposed SiLU pattern (neg, exp, constant, add,
    truediv). Restores the original inner_fn on exit.

    If the pattern is not detected, does nothing.
    """
    if not isinstance(node.data, Pointwise):
        yield
        return

    index_vars = CutlassEVTCodegen.get_index_vars(node)
    recon = _SiLUReconstituter(node.data.inner_fn, index_vars)

    if not recon.detect():
        yield
        return

    replacement_fn = recon.make_replacement_inner_fn()
    orig_inner_fn = node.data.inner_fn
    try:
        object.__setattr__(node.data, "inner_fn", replacement_fn)
        yield
    finally:
        object.__setattr__(node.data, "inner_fn", orig_inner_fn)


class _OpRecorder:
    """Records ops calls and detects the SiLU decomposition pattern.

    SiLU decomposes as: x / (1 + exp(-x))
    The ops sequence is: load → neg → exp → constant(1) → add → truediv
    with to_dtype calls possibly interspersed.

    Returns OpsValue wrappers so that Python arithmetic operators (like -x)
    correctly dispatch through ops.neg(x) etc.
    """

    def __init__(self) -> None:
        self._ops: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []
        self._counter = 0

    def _next_val(self) -> str:
        val = f"_rec_{self._counter}"
        self._counter += 1
        return val

    def __getattr__(self, name: str) -> Any:
        def handler(*args: Any, **kwargs: Any) -> OpsValue:
            val = self._next_val()
            # Unwrap OpsValue args for clean recording
            unwrapped_args = tuple(
                a.value if isinstance(a, OpsValue) else a for a in args
            )
            self._ops.append((name, val, unwrapped_args, kwargs))
            return OpsValue(val)

        return handler

    def _build_dtype_resolver(self) -> tuple[dict[str, str], Any]:
        """Build a mapping from to_dtype outputs to their inputs."""
        dtype_through: dict[str, str] = {}
        for name, val, args, _kwargs in self._ops:
            if name == "to_dtype" and len(args) >= 1:
                dtype_through[val] = str(args[0])

        def resolve(v: str) -> str:
            """Follow to_dtype chain to find the original value."""
            while v in dtype_through:
                v = dtype_through[v]
            return v

        return dtype_through, resolve

    def _get_core_ops(
        self,
    ) -> tuple[
        list[tuple[str, str, tuple[Any, ...]]],
        dict[str, list[tuple[str, str, tuple[Any, ...]]]],
    ]:
        """Get core ops (excluding to_dtype and store) grouped by name."""
        core_ops: list[tuple[str, str, tuple[Any, ...]]] = [
            (name, val, args)
            for name, val, args, _kwargs in self._ops
            if name not in ("to_dtype", "store")
        ]
        by_op: dict[str, list[tuple[str, str, tuple[Any, ...]]]] = {}
        for name, val, args in core_ops:
            by_op.setdefault(name, []).append((name, val, args))
        return core_ops, by_op

    def _verify_silu_dataflow(
        self,
        loads: list[tuple[str, str, tuple[Any, ...]]],
        by_op: dict[str, list[tuple[str, str, tuple[Any, ...]]]],
        resolve: Any,
    ) -> str | None:
        """Verify SiLU data-flow edges and return the buffer name, or None."""
        load_vals = OrderedSet([op[1] for op in loads])
        load_names = OrderedSet([op[2][0] for op in loads])
        if len(load_names) != 1:
            return None

        neg_op = by_op["neg"][0]
        exp_op = by_op["exp"][0]
        const_op = by_op["constant"][0]
        add_op = by_op["add"][0]
        div_op = by_op["truediv"][0]

        neg_input = resolve(str(neg_op[2][0]))
        exp_input = resolve(str(exp_op[2][0]))
        add_inputs = OrderedSet(
            [resolve(str(add_op[2][0])), resolve(str(add_op[2][1]))]
        )
        div_num = resolve(str(div_op[2][0]))
        div_den = resolve(str(div_op[2][1]))

        if not (
            neg_input in load_vals
            and exp_input == neg_op[1]
            and const_op[2][0] == 1
            and add_inputs == OrderedSet([const_op[1], exp_op[1]])
            and div_num in load_vals
            and div_den == add_op[1]
        ):
            return None

        return next(iter(load_names))

    def detect_silu(self) -> _LoadCapture | None:
        """Check if recorded ops form the SiLU decomposition pattern.

        Only matches when the entire inner_fn is a pure SiLU on a single load
        (possibly with to_dtype wrappers). Does not match SiLU as part of a
        larger expression (e.g., silu(x) * y).

        The ops order may vary (e.g., constant before neg), so we match by
        data-flow edges rather than positional order. to_dtype ops are treated
        as transparent wrappers and followed through in the data-flow graph.

        The real inductor decomposition loads the same buffer twice (x appears
        in both numerator and denominator), so we allow 2 loads of the same
        buffer.

        Returns _LoadCapture for the SiLU input if detected, None otherwise.
        """
        _, resolve = self._build_dtype_resolver()
        _core_ops, by_op = self._get_core_ops()

        required = OrderedSet(["load", "neg", "exp", "constant", "add", "truediv"])
        if by_op.keys() != required:
            return None

        loads = by_op["load"]
        if len(loads) not in (1, 2):
            return None
        if any(len(by_op[k]) != 1 for k in required - OrderedSet(["load"])):
            return None

        load_name = self._verify_silu_dataflow(loads, by_op, resolve)
        if load_name is None:
            return None
        return _LoadCapture(load_name)

    def detect_silu_mul(self) -> tuple[_LoadCapture, _LoadCapture] | None:
        """Check if recorded ops form the silu(x) * y pattern.

        Matches: silu(load(buf_a)) * load(buf_b)
        where silu decomposes as x / (1 + exp(-x)).

        Returns (silu_input_capture, mul_input_capture) if detected.
        """
        _, resolve = self._build_dtype_resolver()
        _core_ops, by_op = self._get_core_ops()

        required = OrderedSet(
            ["load", "neg", "exp", "constant", "add", "truediv", "mul"]
        )
        if by_op.keys() != required:
            return None

        loads = by_op["load"]
        if len(loads) not in (2, 3):
            return None
        if any(len(by_op[k]) != 1 for k in required - OrderedSet(["load"])):
            return None

        # Identify which loads belong to the SiLU and which to the mul operand.
        # The truediv result feeds into mul; the other mul operand is a load
        # from a DIFFERENT buffer.
        div_op = by_op["truediv"][0]
        mul_op = by_op["mul"][0]
        mul_inputs = OrderedSet(
            [resolve(str(mul_op[2][0])), resolve(str(mul_op[2][1]))]
        )

        # One mul input should be the truediv result (SiLU output)
        if div_op[1] not in mul_inputs:
            return None

        # The other mul input should trace back to a load of a different buffer
        other_mul_input = (mul_inputs - OrderedSet([div_op[1]])).pop()

        # Find which load the other_mul_input came from
        other_load = None
        for load_op in loads:
            if load_op[1] == other_mul_input:
                other_load = load_op
                break
        if other_load is None:
            return None

        # The remaining loads (excluding other_load) should all be for the
        # SiLU buffer
        silu_loads = [op for op in loads if op is not other_load]
        silu_load_names = OrderedSet([op[2][0] for op in silu_loads])
        if len(silu_load_names) != 1:
            return None

        # Verify SiLU data-flow on the silu_loads
        silu_name = self._verify_silu_dataflow(silu_loads, by_op, resolve)
        if silu_name is None:
            return None

        other_name = other_load[2][0]
        if silu_name == other_name:
            # Both buffers are the same — this isn't the mul pattern
            return None

        return (_LoadCapture(silu_name), _LoadCapture(other_name))


def scaled_mm_evt(
    scale_A_name: str, scale_B_name: str, bias_name: str | None, output_name: str
) -> tuple[list[str], dict[str, Any], str]:
    evt_read_names = [scale_A_name, scale_B_name]
    var_name_to_buffer_name = {n: n for n in [scale_A_name, scale_B_name]}
    var_name_to_buffer_name["D"] = output_name
    var_name_to_buffer_name[_ACCUMULATOR_ARG_NAME] = output_name
    expr = f"accum * {scale_A_name} * {scale_B_name}{linesep}"
    if bias_name:
        expr = f"({expr}) + {bias_name}"
        evt_read_names.append(bias_name)
        var_name_to_buffer_name[bias_name] = bias_name

    evt_py_code = f"def fn(accum, {','.join(evt_read_names)}):{linesep}\
    D = {expr}{linesep}\
    return D{linesep}"

    return evt_read_names, var_name_to_buffer_name, evt_py_code


class CutlassEVTOpsMixIn:
    @staticmethod
    def _infix_bin_op(op: str, a: str, b: str) -> str:
        return f"{a} {op} {b}"

    @staticmethod
    def _prefix_bin_op(op: str, a: str, b: str) -> str:
        return f"{op}({a}, {b})"

    @staticmethod
    def _prefix_un_op(op: str, a: str) -> str:
        return f"{op}({a})"

    @staticmethod
    def to_dtype(
        x: str,
        dtype: Any,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = False,
    ) -> str:
        return x

    @staticmethod
    def constant(value: Any, dtype: Any) -> str:
        return str(value)

    @staticmethod
    def neg(x0: str) -> str:
        # Use subtraction from zero instead of unary minus because the
        # CUTLASS PythonASTFrontend has visit_BinOp but no visit_UnaryOp.
        return f"(0.0 - {x0})"

    @staticmethod
    def silu(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("silu", x0)

    @staticmethod
    def mul(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("*", x0, x1)

    @staticmethod
    def truediv(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("/", x0, x1)

    @staticmethod
    def ge(x0: str, x1: str) -> str:
        raise NotImplementedError

    @staticmethod
    def add(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("+", x0, x1)

    @staticmethod
    def relu(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("relu", x0)

    @staticmethod
    def sigmoid(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("sigmoid", x0)

    @staticmethod
    def sub(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("-", x0, x1)

    @staticmethod
    def tanh(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("tanh", x0)

    @staticmethod
    def exp(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("exp", x0)


class MockCutlassHandler(CutlassEVTOpsMixIn, WrapperHandler):
    """Passthrough handler for cutlass ops, used for running epilogue nodes for memory planning"""


class _AssignmentFormatter(DefaultHandler):
    def __init__(self, parent_handler: "CutlassEVTCodegen"):
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # Handle op dispatch here
        if hasattr(self.parent_handler, name):
            fn = getattr(self.parent_handler, name)
            line = fn(*args, **kwargs)
            if name in ("load", "store"):
                return OpsValue(line)
            else:
                var = self.parent_handler._tmp_var()
                line = DelayReplaceLine(
                    var,
                    lambda: "D"
                    if var == self.parent_handler.last_stored_var_name
                    else var,
                    f"{var} = {line}",
                )
                self.parent_handler.body.writeline(line)
                return OpsValue(var)
        else:
            raise NotImplementedError(name)


class CutlassEVTCodegen(CutlassEVTOpsMixIn):
    """
    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTCodegen.ir_to_evt_python_code(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.
    """

    def __init__(self, accumulator_node_name: str, removed_buffers: OrderedSet[str]):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTCodegen.ir_to_evt_python_code static method.

        Args:
            accumulator_node_name: The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
            epilogue_nodes: The list of scheduler nodes to be fused into the epilogue
        """
        self.accumulator_node_name: str = accumulator_node_name  #
        self.body: IndentedBuffer = IndentedBuffer(1)  # The body buffer for codegen
        self.var_counter: Iterator[int] = itertools.count()
        self.store_name_to_value: dict[str, OpsValue] = (
            dict()
        )  # Aliases for subexpression functors
        self.reads: OrderedSet[str] = OrderedSet([])
        # Used for creating example tensors
        self.var_name_to_buffer_name: dict[str, str] = {
            _ACCUMULATOR_ARG_NAME: accumulator_node_name
        }
        self.removed_buffers: OrderedSet[str] = removed_buffers
        self.cur_node: ComputedBuffer | None = None
        self.name_to_buffer = V.graph.name_to_buffer | V.graph.graph_inputs
        for name in V.graph.constants:
            # pyrefly: ignore [unsupported-operation]
            self.name_to_buffer[name] = V.graph.add_tensor_constant(
                V.graph.constants[name], name
            )
        self.is_D_assigned = False
        self.D_var_name = None

        if accumulator_node_name not in removed_buffers:
            # cannot return accumulator directly, so alias it
            var = self._tmp_var()
            self.body.writeline(f"{var} = {_ACCUMULATOR_ARG_NAME}")
            self.store(accumulator_node_name, value=OpsValue(var))

    @staticmethod
    def ir_to_evt_python_code(
        cutlass_template_node_name: str,
        epilogue_nodes: list[BaseSchedulerNode],
        removed_buffers: OrderedSet[str],
    ) -> tuple[list[str], list[str], dict[str, Any], str]:
        codegen = CutlassEVTCodegen(cutlass_template_node_name, removed_buffers)
        handler = _AssignmentFormatter(codegen)

        with virtualized.V.set_ops_handler(handler):
            for s_node in epilogue_nodes:
                node = s_node.node
                assert isinstance(node, ComputedBuffer)
                with codegen.set_cur_node(node):
                    index_vars = CutlassEVTCodegen.get_index_vars(node)
                    # Reconstitute decomposed SiLU pattern to native silu()
                    # for CUTLASS EVT. See reconstitute_silu() docstring.
                    with reconstitute_silu(node):
                        node.get_store_function()(index_vars)

        codegen.finalize()

        return (
            codegen.get_reads(),
            codegen.get_writes(),
            codegen.get_renames(),
            codegen.get_value(),
        )

    def get_value(self) -> str:
        return linesep.join(
            [
                self._render_input_signature(),
                self.body.getvalue(),
                self._render_return_statement(),
            ]
        )

    def finalize(self) -> None:
        # Rename the last store to D
        # no other code references this store
        # to workaround https://github.com/NVIDIA/cutlass/issues/2288
        # Note: the delayed line will automatically rewrite the last assignment to
        # be to D
        buffer_name = self.var_name_to_buffer_name[self.last_stored_var_name]
        self.var_name_to_buffer_name.pop(self.last_stored_var_name)
        self.var_name_to_buffer_name["D"] = buffer_name
        self.store_name_to_value[buffer_name] = OpsValue("D")

    @contextmanager
    def set_cur_node(self, node: ComputedBuffer) -> Generator[None, Any, Any]:
        prev_node = self.cur_node
        try:
            self.cur_node = node
            yield
        finally:
            self.cur_node = prev_node

    def get_renames(self) -> dict[str, str]:
        return dict(self.var_name_to_buffer_name)

    def get_reads(self) -> list[str]:
        return list(self.reads.difference(self.store_name_to_value.keys()))

    def get_writes(self) -> list[str]:
        return list(self.store_name_to_value.keys())

    def load(self, name: str, index: Any) -> str:
        self._check_indexing(name, index)
        if name in self.store_name_to_value:
            return self.store_name_to_value[name].value
        elif name == self.accumulator_node_name:
            return _ACCUMULATOR_ARG_NAME
        else:
            self.reads.add(name)
            self.var_name_to_buffer_name[name] = name
            return name

    def store(
        self, name: Any, index: Any = None, value: Any = None, mode: Any = None
    ) -> None:
        if name not in self.removed_buffers:
            if index:
                self._check_indexing(name, index)
            assert value.value != _ACCUMULATOR_ARG_NAME, (
                "Cannot store accumulator arg name"
            )
            self.var_name_to_buffer_name[value.value] = name
            self.store_name_to_value[name] = value
            self.last_stored_var_name = value.value
        return None

    def _get_cur_node(self) -> ComputedBuffer:
        assert self.cur_node
        return self.cur_node

    @staticmethod
    def get_index_vars(node: ComputedBuffer) -> Sequence[sympy.Expr]:
        data = node.data
        # TODO mlazos: relax this, cutlass supports reductions and other ops
        assert isinstance(data, Pointwise)
        return data._index(data.ranges)

    def _get_current_index_vars(self) -> Sequence[sympy.Expr]:
        return self.get_index_vars(self._get_cur_node())

    def _check_indexing(self, name: str, index: sympy.Expr) -> None:
        # We only support indexing that matches the layout today because
        # CUTLASS doesn't support arbitrary indexing
        buffer_name = (
            self.accumulator_node_name if name == _ACCUMULATOR_ARG_NAME else name
        )
        buffer = self.name_to_buffer[buffer_name]
        index_strides = V.graph.sizevars.stride_vars(
            index, self._get_current_index_vars()
        )
        stride = buffer.get_layout().stride
        if not self._stride_compatible(stride, index_strides):
            raise NotImplementedError(
                f"Unsupported indexing for {name} with index {index}, index strides {index_strides}, and layout stride {stride}"
            )

    def _stride_compatible(
        self, left: Iterable[sympy.Expr], right: Iterable[sympy.Expr]
    ) -> bool:
        return all(
            sympy.Eq(l, r) or sympy.Eq(l, 0) or sympy.Eq(r, 0)
            for l, r in (zip(left, right))
        )

    def _render_input_signature(self) -> str:
        arguments = ", ".join(
            [_ACCUMULATOR_ARG_NAME]
            + [name for name in self.reads if name != self.accumulator_node_name]
        )
        return f"def fn({arguments}):"

    def _render_return_statement(self) -> str:
        return_vars = OrderedSet(
            op_v.value for op_v in self.store_name_to_value.values()
        )
        assert "D" in return_vars
        return f"return {', '.join(return_vars)}"

    def _tmp_var(self) -> str:
        return f"tmp_{next(self.var_counter)}"
