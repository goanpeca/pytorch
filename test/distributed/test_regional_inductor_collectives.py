# Owner(s): ["module: dynamo"]

import copy
import os
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch._guards import tracing, TracingContext
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.regional_inductor import (
    _functionalize_inplace_collectives,
    regional_inductor,
)
from torch.testing._internal.common_utils import run_tests, TestCase


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _f(t):
    t = t.clone()
    dist.all_reduce(t)
    return t + 1


def _make_fx_with_allreduce():
    return make_fx(_f)(torch.ones(4))


class TestRegionalInductorCollectives(TestCase):
    def setUp(self):
        super().setUp()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29516")
        with tempfile.NamedTemporaryFile(
            prefix="regional_inductor_store_", delete=False
        ) as fd:
            self._store_path = fd.name
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            store=dist.FileStore(self._store_path, 1),
        )

    def tearDown(self):
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass
        if os.path.exists(self._store_path):
            os.unlink(self._store_path)
        super().tearDown()

    def test_functionalize_inplace_allreduce(self):
        gm = _make_fx_with_allreduce()
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    _torchbind_obj1 = self._torchbind_obj1
    allreduce_ = torch.ops.c10d.allreduce_.default([clone], _torchbind_obj0, _torchbind_obj1, None, False);  clone = _torchbind_obj0 = _torchbind_obj1 = None
    getitem = allreduce_[0]
    getitem_1 = getitem[0];  getitem = None
    getitem_2 = allreduce_[1];  allreduce_ = getitem_2 = None
    add = torch.ops.aten.add.Tensor(getitem_1, 1);  getitem_1 = None
    return add""",
        )
        self.assertTrue(any(k.startswith("_torchbind_obj") for k in gm.__dict__))

        _functionalize_inplace_collectives(gm)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', '0')
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, 1);  wait_tensor_default = None
    return add""",
        )
        # Torchbind ProcessGroup / ReduceOp attrs are stripped so downstream
        # consumers can deepcopy the GraphModule.
        self.assertFalse(any(k.startswith("_torchbind_obj") for k in gm.__dict__))
        copy.deepcopy(gm)

        # Numerics: rewritten graph must match eager.
        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(gm(x), _f(x))

    def test_functionalize_inplace_allreduce_compile_on_one_rank(self):
        # Mirrors Dynamo under ``compile_on_one_rank``: the PG is passed as a
        # graph argument (``get_attr`` node) instead of being baked in as the
        # group_name string. The gm attr is left as the original torchbind
        # ScriptObject — the caller is responsible for any unboxing /
        # substitution before execution. Structure-only check; running this
        # graph in-process would need extra setup.
        gm = _make_fx_with_allreduce()
        with dist.config.patch(compile_on_one_rank=True):
            _functionalize_inplace_collectives(gm)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', _torchbind_obj0);  _torchbind_obj0 = None
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, 1);  wait_tensor_default = None
    return add""",
        )
        # PG torchbind attr is retained unchanged (still a ScriptObject); the
        # ReduceOp attr is dropped because its get_attr became dead.
        self.assertTrue(hasattr(gm, "_torchbind_obj0"))
        self.assertIsInstance(gm._torchbind_obj0, torch.ScriptObject)
        self.assertFalse(hasattr(gm, "_torchbind_obj1"))

    @unittest.expectedFailure
    def test_regional_inductor_under_compile_on_one_rank_xfail(self):
        # ``compile_on_one_rank=True`` makes the rewrite keep the PG as a
        # graph arg, but Inductor's comm lowering can't realize a
        # ``TorchBindObject`` (it expects a ``group_name`` string). This is
        # a pre-existing Inductor limitation — the same failure reproduces
        # with stock ``torch.compile + Inductor + compile_on_one_rank=True``
        # and is unrelated to this pass. Marked xfail so the test surfaces
        # if/when Inductor gains lowering support for the PG-passthrough form.
        gm = _make_fx_with_allreduce()
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                    "inductor_configs": {}
                }
        fake_mode = next(
            n.meta["val"].fake_mode
            for n in gm.graph.nodes
            if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
        )

        with (
            dist.config.patch(compile_on_one_rank=True),
            tracing(TracingContext(fake_mode)),
        ):
            compiled_gm = regional_inductor(gm)

        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(compiled_gm([x]), _f(x))

    def test_regional_inductor_with_dist_all_reduce(self):
        # End-to-end check that ``regional_inductor`` (which calls
        # ``_functionalize_inplace_collectives`` internally) handles a
        # ``make_fx`` graph containing ``dist.all_reduce``. Without the
        # rewrite, ``standalone_compile`` would crash deepcopying the
        # ProcessGroup torchbind arg — reaching the assertions at all proves
        # the rewrite ran.
        gm = _make_fx_with_allreduce()
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                    "inductor_configs": {}
                }

        fake_mode = next(
            n.meta["val"].fake_mode
            for n in gm.graph.nodes
            if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
        )

        with tracing(TracingContext(fake_mode)):
            compiled_gm = regional_inductor(gm)

        # The rewritten region (clone + functional all_reduce + wait + copy_ +
        # add) has been scooped into a single standalone-compiled call.
        self.assertExpectedInline(
            compiled_gm.code.strip(),
            """\
def forward(self, args_list):
    args_iter = iter(args_list)
    t_1 = next(args_iter)
    args_list.clear()
    inner = torch__inductor_standalone_compile_inner(t_1);  t_1 = None
    getitem_3 = inner[0];  inner = None
    return getitem_3""",
        )

        # Numerics: rewritten + standalone-compiled graph matches eager.
        # ``regional_inductor`` switches to boxed calling convention; pass
        # inputs as a list (the callee clears it).
        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(compiled_gm([x]), _f(x))


if __name__ == "__main__":
    run_tests()
