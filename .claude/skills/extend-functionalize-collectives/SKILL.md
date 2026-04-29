---
name: extend-functionalize-collectives
description: Add support for a new inplace c10d collective (e.g. broadcast_, _allgather_base_, _reduce_scatter_base_, allgather_, reduce_scatter_, alltoall_, alltoall_base_, reduce_, send, recv_) to the FX-level functionalize-collectives pass in torch/fx/passes/regional_inductor.py. Use when the user wants to extend regional_inductor to handle distributed ops beyond all_reduce, when make_fx-traced graphs containing dist.* APIs hit "Tried to deepcopy object __torch__.torch.classes.c10d.ProcessGroup" errors for ops other than allreduce_, or when the user mentions extending _functionalize_inplace_collectives, _inplace_c10d_rewrites, _emit_collective_chain, _resolve_process_group_arg, _resolve_reduce_op_str, _redirect_inplace_collective_uses, or _rewrite_*_ helpers.
---

# Extend functionalize-collectives pass

This skill teaches Claude how to add a new inplace c10d collective to the
functionalize-collectives FX pass at
`torch/fx/passes/regional_inductor.py:_functionalize_inplace_collectives`.

The pass mirrors what Dynamo's `traceable_collective_remaps` does at trace
time, but operates on graphs already produced by `make_fx`. It rewrites
opaque `torch.ops.c10d.{op}_` calls into
`_c10d_functional.{op}` + `wait_tensor` + `aten.copy_` form because
**Inductor's collective machinery only understands the functional form**.
`_inductor.comm_lowering`, `fx_passes.bucketing`, `fx_passes.ddp_fusion`,
`fx_passes.overlap_scheduling`, `fx_passes.micro_pipeline_tp`, and
`fx_passes.low_contention_collectives` pattern-match exclusively on
`torch.ops._c10d_functional.*` + `wait_tensor`. Legacy
`torch.ops.c10d.{op}_` ops have no lowering registered, so a graph that
retains them either crashes at the lowering phase or skips every
comm/compute optimization. The compute/comm overlap scheduler also needs
a standalone `wait_tensor` node to schedule against — there is no
equivalent for the opaque `Work` handle returned by the inplace ops.

## When to use this skill

- The user asks to support a new collective in `regional_inductor`, e.g.
  "add `broadcast_` support", "make `_allgather_base_` work after make_fx".
- A `make_fx`-traced graph crashes downstream with
  `Tried to deepcopy object __torch__.torch.classes.c10d.ProcessGroup` for
  an op other than `allreduce_`.
- The user mentions any of the helpers: `_functionalize_inplace_collectives`,
  `_inplace_c10d_rewrites`, `_emit_collective_chain`,
  `_resolve_process_group_arg`, `_resolve_reduce_op_str`,
  `_redirect_inplace_collective_uses`, or wants to add a new
  `_rewrite_<op>_`.

## Architecture

The pass is built from small composable pieces. Per-op rewrites only need to
glue them together — they don't reinvent any plumbing.

```
_functionalize_inplace_collectives(gm) -> gm        # outer driver
    │
    ├── _inplace_c10d_rewrites()                    # dispatch dict
    │
    ├── _rewrite_<op>_(gm, node)                    # per-op rewrite
    │     ├── _resolve_process_group_arg(gm, arg)  # PG  -> str
    │     ├── _resolve_reduce_op_str(gm, arg)       # ReduceOp -> "sum"/...
    │     ├── _emit_collective_chain(...)           # functional + wait + copy_
    │     └── _redirect_inplace_collective_uses(    # any (..., Work) schema
    │           gm, node, wait_nodes               #   pass list of waits;
    │         )                                    #   helper auto-detects
    │                                              #   Tensor[] vs Tensor
    │
    └── final cleanup: erase rewritten c10d node, drop unused get_attr nodes
                       and their backing torchbind attrs
```

Three things are shared across all rewrites:

1. **Arg resolution.** `_resolve_process_group_arg(gm, arg)` returns
   `pg.group_name` (unboxes the torchbind ScriptObject if needed).
   `_resolve_reduce_op_str(gm, arg)` converts a `ReduceOp` ScriptObject (or
   enum) to `"sum"`/`"avg"`/etc.
2. **Chain construction.** `_emit_collective_chain(gm, before, input_t,
   output_t, target, extra_args, group_name, custom)` inserts
   `target(input, *extra, group_name)` -> `wait_tensor` -> `aten.copy_(output,
   wait)` and propagates `meta["val"]` (using `output_t`'s fake tensor) and
   `meta["custom"]`. Returns the wait node.
3. **Use redirection.** A single helper —
   `_redirect_inplace_collective_uses(gm, node, wait_nodes)` — handles both
   `(Tensor[], Work)` and `(Tensor, Work)` output shapes. Always pass the
   wait nodes as a list (singleton list for the `Tensor` schema).

## Output schemas

Inplace c10d ops fall into three groups:

| Output schema       | Example ops                                                                | How to redirect                                                       |
|---------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `(Tensor[], Work)`  | `allreduce_`, `broadcast_`, `reduce_scatter_`, `alltoall_`                  | `_redirect_inplace_collective_uses(gm, node, [wait_0, wait_1, ...])`  |
| `(Tensor, Work)`    | `_allgather_base_`, `_reduce_scatter_base_`                                | `_redirect_inplace_collective_uses(gm, node, [wait])` (singleton)    |
| `Work` (no tensor)  | `alltoall_base_`, `reduce_`, `send`, `recv_`                                | (none — the op is pure side-effect; `aten.copy_` already updates the input/output tensor in place) |

`_redirect_inplace_collective_uses` auto-detects whether the tensor slot is
a list or a scalar by looking at the user chain of `getitem(node, 0)` — no
schema flag is needed. For `Work`-only ops there is no tensor consumer to
redirect, so just call `_emit_collective_chain` (the inplace `aten.copy_`
captures the side-effect) and skip the redirect helper entirely.

## Common arg patterns

Determining inputs/outputs and extra args by op family:

| c10d op                  | Input tensor(s)   | Output tensor(s)   | Functional target                                  | Extra args                                                |
|--------------------------|-------------------|--------------------|----------------------------------------------------|-----------------------------------------------------------|
| `allreduce_`             | `tensors[i]`      | `tensors[i]` (same) | `_c10d_functional.all_reduce`                      | `(op_str,)`                                               |
| `broadcast_`             | `tensors[i]`      | `tensors[i]` (same) | `_c10d_functional.broadcast`                       | `(root_rank,)`                                            |
| `_allgather_base_`       | `args[1]` (input) | `args[0]` (output) | `_c10d_functional.all_gather_into_tensor`          | `(group_size,)` from `pg.size()`                          |
| `_reduce_scatter_base_`  | `args[1]`         | `args[0]`          | `_c10d_functional.reduce_scatter_tensor`           | `(op_str, group_size)`                                    |
| `allgather_`             | per input list    | concat output[][]  | `_c10d_functional.all_gather_into_tensor` + slices | per-input group_size, then split output back              |
| `reduce_scatter_`        | concat input[][]  | per output list    | `_c10d_functional.reduce_scatter_tensor`           | `(op_str, group_size)` per element                        |

Look at the Python wrappers in `torch/distributed/_functional_collectives.py`
(e.g. `all_reduce_inplace`, `all_gather_tensor_inplace`,
`reduce_scatter_tensor_inplace`) for an authoritative source of how each
inplace API maps to the functional collective + wait + copy_ pattern. This
pass mirrors those wrappers at the FX level.

## How to add a new op

### Step 1 — Inspect the schema

```python
print(torch.ops.c10d.<op>_.default._schema)
```

Note positions of: tensor list / output / input args, ProcessGroup arg,
ReduceOp arg (if any), and the output tuple shape.

### Step 2 — Write `_rewrite_<op>_(gm, node)`

Two templates cover almost every case.

**Template A — `(Tensor[], Work)` schema** (per-tensor rewrite, e.g. `broadcast_`):

```python
def _rewrite_broadcast_(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Schema: c10d::broadcast_(Tensor[] tensors, ProcessGroup pg,
    int root_rank, int root_tensor, bool async_op, int timeout=-1)
    -> (Tensor[], Work)
    """
    tensors = node.args[0]
    group_name = _resolve_process_group_arg(gm, node.args[1])
    root_rank = node.args[2]
    custom = node.meta.get("custom")
    target = torch.ops._c10d_functional.broadcast.default
    waits = [
        _emit_collective_chain(gm, node, t, t, target, (root_rank,), group_name, custom)
        for t in tensors
    ]
    _redirect_inplace_collective_uses(gm, node, waits)
```

**Template B — `(Tensor, Work)` schema with separate output/input** (e.g.
`_allgather_base_`, `_reduce_scatter_base_`):

```python
def _rewrite__allgather_base_(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Schema: c10d::_allgather_base_(Tensor output, Tensor input,
    ProcessGroup pg, bool async_op, int timeout=-1) -> (Tensor, Work)
    """
    import torch.distributed as dist

    output_t, input_t = node.args[0], node.args[1]
    pg_arg = node.args[2]
    group_name = _resolve_process_group_arg(gm, pg_arg)
    # Need pg.size() too — re-resolve to grab the unboxed Python wrapper.
    pg = getattr(gm, pg_arg.target) if isinstance(pg_arg, torch.fx.Node) else pg_arg
    if isinstance(pg, torch.ScriptObject):
        pg = dist.ProcessGroup.unbox(pg)
    group_size = pg.size()
    custom = node.meta.get("custom")
    target = torch.ops._c10d_functional.all_gather_into_tensor.default
    wait = _emit_collective_chain(
        gm, node, input_t, output_t, target, (group_size,), group_name, custom
    )
    _redirect_inplace_collective_uses(gm, node, [wait])  # singleton list
```

Notes:
- `_emit_collective_chain` copies `output_t.meta["val"]` onto every new
  node. That works for ops where the functional collective produces a tensor
  with the same shape/dtype as `output_t`. For most c10d collectives this
  holds (the inplace c10d op was already storing into `output_t`, and the
  functional collective produces the same layout). If a future op needs a
  different intermediate shape, materialize the val under the input's
  fake_mode and set `meta["val"]` manually instead of relying on the helper.

### Step 3 — Register in `_inplace_c10d_rewrites`

```python
def _inplace_c10d_rewrites() -> dict[Any, _InplaceCollectiveRewrite]:
    if not torch.distributed.is_available():
        return {}
    return {
        torch.ops.c10d.allreduce_.default: _rewrite_allreduce_,
        torch.ops.c10d.<op>_.default: _rewrite_<op>_,   # <-- new
    }
```

That is the entire wiring. The driver discovers the new rewrite, calls it,
erases the rewritten node, and runs cleanup automatically.

### Step 4 — Add a test

Add a case to `test/distributed/test_regional_inductor_collectives.py`. The
file uses single-rank Gloo (CPU, in-process via `FileStore`) so the rewrite
can be exercised against real `dist.*` numerics — with `world_size=1` the
collective is the identity, so any deviation in the rewrite shows up as a
mismatch against eager.

Prefer `assertExpectedInline` over `FileCheck` so structural drift in the
rewritten graph (op order, dead-code cleanup, arg form) fails loudly with a
readable diff — see `test_functionalize_inplace_allreduce` for the pattern.

```python
def test_functionalize_inplace_<op>(self):
    def f(t):
        t = t.clone()
        dist.<op>(t, ...)        # the dist.* API that lowers to c10d.<op>_
        return t + 1

    gm = make_fx(f)(torch.ones(...))
    self.assertExpectedInline(gm.code.strip(), """\
... pre-rewrite snapshot ...""")

    gm = _functionalize_inplace_collectives(gm)

    self.assertExpectedInline(gm.code.strip(), """\
... post-rewrite snapshot ...""")

    copy.deepcopy(gm)            # must succeed: no torchbind attrs left

    # Numerics: rewritten graph must match eager.
    x = torch.arange(..., dtype=torch.float32)
    self.assertEqual(gm(x), f(x))
```

Tip: write the test with placeholder strings, run once with
`EXPECTTEST_ACCEPT=1 python test/distributed/test_regional_inductor_collectives.py`
to capture the actual graph snapshots, then re-run normally to confirm.

Run:
```bash
python test/distributed/test_regional_inductor_collectives.py
```

### Step 5 — Lint

```bash
lintrunner torch/fx/passes/regional_inductor.py test/distributed/test_regional_inductor_collectives.py
lintrunner -a torch/fx/passes/regional_inductor.py test/distributed/test_regional_inductor_collectives.py  # autofix
```

## Common pitfalls

- **Reading `pg.group_name` on the raw ScriptObject fails.** The torchbind
  `ProcessGroup` exposes only `__init__`. Use `_resolve_process_group_arg`
  (or `dist.ProcessGroup.unbox(pg)` if you need other PG state like
  `pg.size()`).
- **Treating `dist.ReduceOp.SUM` and the torchbind `ReduceOp` as
  interchangeable.** They are not. Always go through `_resolve_reduce_op_str`,
  which handles both forms.
- **Forgetting `meta["val"]` on the new nodes** breaks the partitioner with
  `Partition is bad because non fake tensor value is seen ...`.
  `_emit_collective_chain` propagates it for you; only override when the
  output shape genuinely differs from `output_t`.
- **Forgetting `meta["custom"]`** drops the `compile_with_inductor`
  annotation, so the new functional ops fall outside any inductor region
  and the partitioner may split awkwardly. `_emit_collective_chain` carries
  it forward automatically.
- **Not erasing the original c10d node.** `eliminate_dead_code` treats
  inplace c10d ops as impure (they are), so it will not remove them. The
  driver erases via `gm.graph.erase_node(node)` after the rewrite runs —
  per-op rewrites should NOT erase `node` themselves.
- **Leaving torchbind attrs on the GraphModule.** Even after the `get_attr`
  node is gone, `gm._torchbind_obj0` itself remains in `gm.__dict__` and
  re-triggers the deepcopy crash. The driver loops over remaining
  `get_attr` nodes with no users and `delattr`s them; verify with:
  ```python
  assert not [k for k in gm.__dict__ if k.startswith("_torchbind_obj")]
  ```
- **Forgetting that the driver returns `gm`.** `_functionalize_inplace_collectives`
  returns the (possibly mutated) GraphModule to match the
  `_create_inductor_marked_regions` / `_compile_inductor_marked_regions`
  pattern. Use `gm = _functionalize_inplace_collectives(gm)` when chaining.

## Reference

- Pass driver, helpers, `_rewrite_allreduce_`:
  `torch/fx/passes/regional_inductor.py` — search for
  `--- functionalize-collectives pass ---`.
- Inplace c10d schemas: `torch/csrc/distributed/c10d/Ops.cpp`.
- Functional collective wrappers (best source for arg derivation):
  `torch/distributed/_functional_collectives.py` — see
  `all_reduce_inplace`, `all_gather_tensor_inplace`,
  `reduce_scatter_tensor_inplace`, `all_to_all_inplace`.
- Dynamo's parallel rewrite at trace time: search for
  `traceable_collective_remaps` in
  `torch/distributed/_functional_collectives.py` and
  `CollectiveFunctionRewriteVariable` in
  `torch/_dynamo/variables/functions.py`.
- Existing test file: `test/distributed/test_regional_inductor_collectives.py`.
