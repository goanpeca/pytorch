import contextlib
import functools
import logging
import operator
from collections.abc import Callable, Iterator, Mapping
from typing import Any, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")

import torch
from torch.fx._compatibility import compatibility


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor"]


# --- functionalize-collectives pass --------------------------------------
# Shared scaffolding for rewriting raw ``c10d.{op}_`` inplace ops produced by
# ``make_fx`` into ``_c10d_functional.{op}`` + ``wait_tensor`` + ``aten.copy_``
# form. The structure separates concerns:
#   * Arg resolution (PG, ReduceOp) lives in ``_resolve_*``.
#   * The functional + wait + copy_ chain construction lives in
#     ``_emit_collective_chain`` (used by every per-op rewrite).
#   * Use redirection lives in ``_redirect_*_work_uses`` keyed by output schema.
#   * Per-op rewrites (``_rewrite_<op>_``) glue these together.
#   * The driver ``_functionalize_inplace_collectives`` walks the graph,
#     dispatches by target, and runs final cleanup.
# Adding a new op = one ``_rewrite_<op>_`` + one entry in
# ``_inplace_c10d_rewrites``.


def _resolve_process_group_arg(
    gm: torch.fx.GraphModule, arg: torch.fx.Node
) -> str | torch.fx.Node:
    """Resolve the ``group`` arg for a functional collective call by simulating
    what would happen inside the functional collective wrapper if Dynamo had
    traced it.

    Delegates to ``torch.distributed._functional_collectives._group_or_group_name``
    so this pass produces the same shape Dynamo's
    ``CollectiveFunctionRewriteVariable`` -> ``all_reduce_inplace`` -> ``all_reduce``
    chain would have produced, including any flag-driven behavior
    (``compile_on_one_rank``, etc.). In the string-returning case we use the
    string directly; in the PG-returning case we hand back the original
    ``get_attr`` Node so the PG flows as a live graph value (the caller owns
    any further unboxing / substitution before execution).
    """
    import torch.distributed as dist
    from torch.distributed._functional_collectives import _group_or_group_name

    if arg.op != "get_attr":
        raise AssertionError(f"expected get_attr, got op={arg.op!r}")
    pg = getattr(gm, arg.target)  # type: ignore[arg-type]
    if isinstance(pg, torch.ScriptObject):
        pg = dist.ProcessGroup.unbox(pg)
    resolved = _group_or_group_name(pg)
    return arg if isinstance(resolved, dist.ProcessGroup) else resolved


def _resolve_reduce_op_str(gm: torch.fx.GraphModule, arg: torch.fx.Node) -> str:
    """Get the lower-case op string (``"sum"``/``"avg"``/...) for a c10d
    ReduceOp ``get_attr`` arg, converting from the torchbind ScriptObject
    form if needed.
    """
    import torch.distributed as dist
    from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

    if arg.op != "get_attr":
        raise AssertionError(f"expected get_attr, got op={arg.op!r}")
    reduce_op = getattr(gm, arg.target)  # type: ignore[arg-type]
    if isinstance(reduce_op, torch.ScriptObject):
        reduce_op = dist.ReduceOp.RedOpType(reduce_op.op())  # type: ignore[attr-defined]
    return REDUCE_OP_TO_STR[reduce_op]


def _emit_collective_chain(
    gm: torch.fx.GraphModule,
    before: torch.fx.Node,
    input_t: torch.fx.Node | Any,
    output_t: torch.fx.Node | Any,
    functional_target: Any,
    extra_args: tuple[Any, ...],
    group_arg: str | torch.fx.Node,
    custom: Any,
) -> torch.fx.Node:
    """Insert ``_c10d_functional.<op>(input, *extra_args, group_arg)`` ->
    ``wait_tensor`` -> ``aten.copy_(output, wait)`` immediately before
    ``before`` and return the ``wait_tensor`` node.

    ``group_arg`` is either a ``group_name`` string (deepcopy-safe form,
    Dynamo's default) or a ``get_attr`` Node referencing the PG (mirrors
    Dynamo under ``compile_on_one_rank``).

    Propagates ``output_t.meta["val"]`` onto every new node — every functional
    collective we rewrite to produces a tensor with the same shape/dtype as
    ``output_t``, and ``aten.copy_`` returns ``output_t``. Also carries the
    rewritten node's ``custom`` annotation forward so region scooping continues
    to work.
    """
    func_ops = torch.ops._c10d_functional
    val = output_t.meta.get("val") if isinstance(output_t, torch.fx.Node) else None
    with gm.graph.inserting_before(before):
        ar = gm.graph.call_function(
            functional_target, (input_t, *extra_args, group_arg)
        )
        wait = gm.graph.call_function(func_ops.wait_tensor.default, (ar,))
        copy_ = gm.graph.call_function(torch.ops.aten.copy_.default, (output_t, wait))
        if val is not None:
            ar.meta["val"] = wait.meta["val"] = copy_.meta["val"] = val
        if custom is not None:
            ar.meta["custom"] = wait.meta["custom"] = copy_.meta["custom"] = custom
    return wait


def _redirect_inplace_collective_uses(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    wait_nodes: list[torch.fx.Node],
) -> None:
    """Re-route uses for an inplace c10d op with output schema
    ``(Tensor[], Work)`` — e.g. ``allreduce_``, ``broadcast_``,
    ``reduce_scatter_``, ``alltoall_``.

    ``make_fx`` lowers tensor consumers into ``getitem(node, 0)[i]``; replace
    each with ``wait_nodes[i]``. The work-handle ``getitem(node, 1)`` is erased
    if unused (live users raise — functional collectives have no Work object
    to forward to, so we can't preserve that synchronization).

    TODO: extend to ``(Tensor, Work)`` schemas (``_allgather_base_``,
    ``_reduce_scatter_base_``) when those rewrites are added. There the
    ``getitem(node, 0)`` is itself the tensor consumer (no inner
    ``getitem`` chain), so the inner loop below would need to fall back to
    ``use.replace_all_uses_with(wait_nodes[0])`` when ``use`` has no inner
    ``getitem`` users.
    """
    for use in list(node.users):
        if use.op != "call_function" or use.target is not operator.getitem:
            raise AssertionError(
                f"Cannot functionalize {node.target}: unexpected use {use} "
                f"(op={use.op!r}, target={use.target!r}). The c10d inplace "
                f"output is a tuple consumed via ``getitem`` in ``make_fx`` "
                f"graphs; non-getitem users are not supported by this pass."
            )
        if use.args[1] == 0:
            for sub_use in list(use.users):
                if (
                    sub_use.op != "call_function"
                    or sub_use.target is not operator.getitem
                ):
                    raise AssertionError(
                        f"Cannot functionalize {node.target}: tensor list "
                        f"consumed by non-getitem user {sub_use}. Extend "
                        f"_redirect_inplace_collective_uses to support this "
                        f"schema if needed."
                    )
                sub_use.replace_all_uses_with(wait_nodes[sub_use.args[1]])  # type: ignore[index]
                gm.graph.erase_node(sub_use)
            gm.graph.erase_node(use)
        elif use.args[1] == 1:
            if use.users:
                raise AssertionError(
                    f"Cannot functionalize {node.target}: Work handle has "
                    f"{len(use.users)} live use(s) ({list(use.users)}). "
                    f"Functional collectives express synchronization via "
                    f"``wait_tensor`` on the data tensor, so the Work object "
                    f"has no equivalent to forward to."
                )
            gm.graph.erase_node(use)


def _rewrite_allreduce_(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Reference rewrite — use this as a template for new collectives.

    Schema: ``c10d::allreduce_(Tensor[] tensors, ProcessGroup pg, ReduceOp op,
    Tensor? sparse_indices, bool async_op, int timeout=-1) -> (Tensor[], Work)``.

    Before::

        allreduce_ = c10d.allreduce_([t0, t1], pg, reduce_op, None, False)
        getitem    = allreduce_[0]
        getitem_1  = getitem[0]      # consumer of t0 post-allreduce
        getitem_2  = getitem[1]      # consumer of t1 post-allreduce
        ...        = allreduce_[1]   # work handle (unused)

    After (per input tensor)::

        ar_i = _c10d_functional.all_reduce(t_i, op_str, group_name)
        wait_i = _c10d_functional.wait_tensor(ar_i)
        _ = aten.copy_(t_i, wait_i)  # preserves inplace semantics
        # downstream uses of ``getitem_<i>`` are redirected to ``wait_i``
    """
    tensors = node.args[0]
    group_arg = _resolve_process_group_arg(gm, node.args[1])  # type: ignore[arg-type]
    op_str = _resolve_reduce_op_str(gm, node.args[2])  # type: ignore[arg-type]
    custom = node.meta.get("custom")
    target = torch.ops._c10d_functional.all_reduce.default
    waits = [
        _emit_collective_chain(gm, node, t, t, target, (op_str,), group_arg, custom)
        for t in tensors  # type: ignore[union-attr]
    ]
    _redirect_inplace_collective_uses(gm, node, waits)


_InplaceCollectiveRewrite = Callable[[torch.fx.GraphModule, torch.fx.Node], None]


def _inplace_c10d_rewrites() -> dict[Any, _InplaceCollectiveRewrite]:
    """Map inplace ``c10d.{op}_`` op targets to per-op rewrite functions.

    To support a new collective: write ``_rewrite_<op>_`` (typically a few
    lines using ``_resolve_process_group_arg`` / ``_resolve_reduce_op_str`` /
    ``_emit_collective_chain`` / ``_redirect_inplace_collective_uses``), then
    register it here.
    """
    if not torch.distributed.is_available():
        return {}
    return {
        torch.ops.c10d.allreduce_.default: _rewrite_allreduce_,
    }


def _functionalize_inplace_collectives(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Rewrite raw ``torch.ops.c10d.{op}_`` inplace calls in ``gm`` into
    ``_c10d_functional.{op}`` + ``wait_tensor`` + ``aten.copy_`` form, mirroring
    how Dynamo rewrites ``torch.distributed.*`` APIs at trace time.

    ``make_fx`` traces ``dist.*`` collectives as opaque ``c10d.{op}_`` calls
    that carry a ProcessGroup torchbind argument. Downstream consumers like
    ``regional_inductor`` -> ``standalone_compile`` deepcopy the graph and fail
    on the non-deepcopiable ProcessGroup. After this pass the graph carries
    only ``_c10d_functional`` ops with string group names, matching the output
    of Dynamo capture.
    """
    rewrites = _inplace_c10d_rewrites()
    if not rewrites:
        return gm

    found = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        rewrite = rewrites.get(node.target, None)
        if rewrite is None:
            continue
        rewrite(gm, node)
        # Inplace c10d ops are impure; ``eliminate_dead_code`` keeps them
        # alive even with no users, so erase the rewritten node explicitly.
        gm.graph.erase_node(node)
        found = True

    if not found:
        return gm

    # Strip orphan ``get_attr`` nodes (typically torchbind ProcessGroup /
    # ReduceOp attrs left behind by the rewrite) along with their backing
    # module attributes so the graph is deepcopy-safe for downstream
    # consumers like ``standalone_compile``.
    from torch.fx.graph_module import _del_attr

    for n in list(gm.graph.find_nodes(op="get_attr")):
        if n.users:
            continue
        _del_attr(gm, n.target)  # type: ignore[arg-type]
        gm.graph.erase_node(n)
    gm.recompile()
    return gm


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return fn(*args, **kwargs)

    return inner


@contextlib.contextmanager
def _disable_remat_for_regional_subcompile() -> Iterator[None]:
    # In torch.compile, regional_inductor subcompiles run after the enclosing
    # non-strict full graph has already been partitioned, so any graph-SAC
    # remat pass has already run before we reach this nested compile.
    # Rerunning remat here can see stage-2-reordered backward nodes that
    # violate remat's contiguous-backward-region assumption.
    with torch._functorch.config.patch(remat_using_tags_for_fwd_loss_bwd_graph=False):
        yield


def _compile_submod(gm: torch.fx.GraphModule, prefix: str) -> torch.fx.GraphModule:
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            submod = getattr(gm, node.target)

            # Get inductor configs from annotation
            # TODO we should change partition when there are multiple differently
            # annotated regions.
            inductor_options: dict[str, Any] = {}
            for sub_node in submod.graph.nodes:
                if hasattr(sub_node, "meta") and sub_node.meta.get("custom", None):
                    custom = sub_node.meta["custom"]
                    if isinstance(custom, dict) and "compile_with_inductor" in custom:
                        compile_value = custom["compile_with_inductor"]
                        if (
                            isinstance(compile_value, dict)
                            and "inductor_configs" in compile_value
                        ):
                            inductor_options = compile_value["inductor_configs"]
                            break

            # Log the options being used
            logger.info(
                "Compiling submodule %s with inductor options: %s",
                node.target,
                inductor_options,
            )

            # Apply config patches before compilation
            import torch._inductor.config as inductor_config

            # Validate that all config keys exist
            for key in inductor_options:
                if not hasattr(inductor_config, key):
                    raise ValueError(
                        f"Invalid inductor config key '{key}' in regional_inductor annotation. "
                        f"Available config keys can be found in torch._inductor.config"
                    )

            with (
                inductor_config.patch(inductor_options),
                _disable_remat_for_regional_subcompile(),
            ):
                compiled_fn = torch._inductor.standalone_compile(
                    submod,
                    fake_inputs,
                    dynamic_shapes="from_tracing_context",
                    aot=True,
                )
            if not isinstance(compiled_fn, AOTCompiledArtifact):
                raise AssertionError(
                    f"Expected AOTCompiledArtifact, got {type(compiled_fn)}"
                )
            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(compiled_fn)
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node) -> bool:
    return bool(
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


class _RegionScooper:
    """
    Scoops out the inductor marked regions. It does NOT compile them.
    """

    @staticmethod
    def scoop_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        from torch.fx.passes.operator_support import create_op_support
        from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

        # Group tagged nodes by region ID.  The region ID comes from the
        # optional "inductor_region" key inside the compile_with_inductor
        # annotation. When absent, all tagged nodes share a single default region
        _DEFAULT_REGION = object()
        regions: dict[object, set[torch.fx.Node]] = {}
        for node in gm.graph.nodes:
            if _needs_inductor_compile(node):
                compile_value = node.meta["custom"]["compile_with_inductor"]
                if (
                    isinstance(compile_value, dict)
                    and "inductor_region" in compile_value
                ):
                    rid = compile_value["inductor_region"]
                else:
                    rid = _DEFAULT_REGION
                regions.setdefault(rid, set()).add(node)

        if not regions:
            logger.info("No inductor marked nodes found")
            return gm

        # Run CapabilityBasedPartitioner per region to get cycle-safe partitions
        # without merging across region boundaries.
        def _is_in_region(
            region_nodes: set[torch.fx.Node],
        ) -> Callable[[Mapping[str, torch.nn.Module], torch.fx.Node], bool]:
            def is_node_supported(
                _submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
            ) -> bool:
                return node in region_nodes

            return is_node_supported

        all_partitions: list[dict[torch.fx.Node, int | None]] = []
        for region_nodes in regions.values():
            support = create_op_support(_is_in_region(region_nodes))
            partitioner = CapabilityBasedPartitioner(
                gm, support, allows_single_node_partition=True
            )
            for partition in partitioner.propose_partitions():
                all_partitions.append(partition.nodes)

        return fuse_by_partitions(
            gm,
            all_partitions,
            prefix="__marked_inductor_submod",
            always_return_tuple=True,
        )

    @staticmethod
    def recursively_scoop_regions(
        gm: torch.fx.GraphModule, _processed: set[int] | None = None
    ) -> torch.fx.GraphModule:
        if _processed is None:
            _processed = set()
        for node in gm.graph.find_nodes(op="get_attr"):
            if _needs_inductor_compile(node):
                # If the get_attr itself is marked for compile, the outer graph will
                # take care of it. If we dont do that, we end up with nested
                # regional inductor compiles that do not work well.
                continue
            submod = getattr(gm, node.target)
            # Track by id: multiple get_attr nodes may reference the same GraphModule
            if (
                isinstance(submod, torch.fx.GraphModule)
                and id(submod) not in _processed
            ):
                _processed.add(id(submod))
                _RegionScooper.recursively_scoop_regions(submod, _processed)

        return _RegionScooper.scoop_regions(gm)

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionScooper.recursively_scoop_regions(gm)


class _RegionCompiler:
    """
    Compiles the scooped out regions.
    """

    @staticmethod
    def compile_region(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch.fx.graph import _BoxedCodeGen

        gm = _compile_submod(gm, "__marked_inductor_submod")
        gm.graph.set_codegen(_BoxedCodeGen())
        gm.recompile()
        return gm

    @staticmethod
    def recursively_compile_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # Find if the graph module has a scooped out region
        found_region = False
        for node in gm.graph.find_nodes(op="call_module"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                if node.target.startswith("__marked_inductor_submod"):
                    found_region = True

        # Recurse through the subgraphs
        for node in gm.graph.find_nodes(op="get_attr"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                _RegionCompiler.recursively_compile_regions(submod)

        if found_region:
            return _RegionCompiler.compile_region(gm)
        return gm

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionCompiler.recursively_compile_regions(gm)


def _create_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionScooper()(gm)


def _compile_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionCompiler()(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor(
    gm: torch.fx.GraphModule, *example_args: object
) -> torch.fx.GraphModule:
    """
    Scoops out inductor marked regions and compiles them with inductor.

    Inductor options should be provided via the annotation API::

        with fx_traceback.annotate(
            {
                "compile_with_inductor": {
                    "inductor_configs": {
                        "max_autotune": True,
                        "triton.cudagraphs": False,
                    }
                }
            }
        ):
            ...
    """

    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues

    with torch.fx.traceback.preserve_node_meta(enable=False):
        # Inplace c10d collectives (e.g. produced by ``make_fx`` over
        # ``dist.all_reduce``) carry non-deepcopiable ProcessGroup torchbind
        # args; rewrite them to ``_c10d_functional`` form first so that
        # ``standalone_compile`` (which deepcopies the submodule) succeeds.
        #
        # Note: ``compile_on_one_rank=True`` produces the PG-passthrough form
        # (``_c10d_functional.<op>(t, ..., <pg>)``), which Inductor's
        # collective lowerings cannot currently realize (they expect a
        # ``group_name`` string). The combination will fail downstream in
        # Inductor with ``NotImplementedError: realize NYI on TorchBindObject``;
        # this is a pre-existing Inductor limitation, not specific to this
        # pass. Use ``compile_on_one_rank=False`` (the default) until
        # Inductor's lowering gains support for the PG-passthrough form.
        gm = _functionalize_inplace_collectives(gm)
        gm = _create_inductor_marked_regions(gm)
        gm = _compile_inductor_marked_regions(gm)
        if torch._functorch.config.force_autograd_cache:
            from torch._inductor.output_code import RegionalOutputCode

            return RegionalOutputCode(gm)  # type: ignore[return-value]
        return gm
