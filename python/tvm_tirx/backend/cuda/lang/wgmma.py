# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Helpers for Hopper (sm_90a) warpgroup MMA: ``wgmma.mma_async``.

The PTX table already spells every WGMMA instruction; these helpers own the
parts kernels get wrong:

* the **chain protocol**: fence the operand registers, ``wgmma.fence``, issue the
  k-step MMAs back to back, ``commit_group``, and later ``wait_group`` and fence
  again. Any non-wgmma access to an accumulator between the fence and the wait
  makes ptxas serialize the chain (remark C7511/C7514, surfaced by
  ``backend_config={"cuda": {"perf_remarks": "error"}}``);
* **shared-memory descriptors** for the canonical swizzled layouts, K-major and
  MN-major, and their k-step advance;
* the **register fragment maps** of ISA 9.7.17.5.1: the m64nN accumulator, and
  the m64kK A operand of the register (``rs``) form. The first 16 columns of an
  f32 accumulator, packed in pairs, are exactly an rs A fragment, which is how
  attention feeds softmax(QK^T) into PV.

All helpers are plain Python that emit TIR while a ``@T.function`` body is being
traced. Call them from inside the body.
"""

from __future__ import annotations

import dataclasses

from tvm_tirx.script import tirx as T

# PTX ISA 9.7.17.2: K per instruction is 32 bytes of A/B.
_K_BYTES = 32
_PTX_TYPE = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float8_e4m3fn": "e4m3",
    "float8_e5m2": "e5m2",
}
_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1}
# Descriptor layout codes taken by cuda.wgmma_encode_matrix_descriptor's
# `swizzle` argument (it maps them to the GMMA bitfield: 0 -> none, 1 -> 32B,
# 2 -> 64B, 3 -> 128B).
_SWIZZLE_CODE = {0: 0, 32: 1, 64: 2, 128: 3}


@dataclasses.dataclass(frozen=True)
class WGMMA:
    """One ``wgmma.mma_async`` shape, m64nNkK, for one warpgroup (128 threads).

    Parameters
    ----------
    n : int
        N of the instruction: a multiple of 8, from 8 to 256.
    ab_dtype : str
        Element type of A and B: ``float16``, ``bfloat16``, ``float8_e4m3fn`` or
        ``float8_e5m2``. ``b_dtype`` may differ for fp8 (e4m3 x e5m2).
    acc_dtype : str
        ``float32`` or ``float16``.
    a_in_registers : bool
        The ``rs`` form: A comes from registers (see :func:`a_fragment_coord`).
    """

    n: int
    ab_dtype: str = "bfloat16"
    acc_dtype: str = "float32"
    a_in_registers: bool = False
    b_dtype: str | None = None

    def __post_init__(self):
        if self.n % 8 or not 8 <= self.n <= 256:
            raise ValueError(f"WGMMA n={self.n} must be a multiple of 8 in [8, 256]")
        for dtype in (self.ab_dtype, self.b_type):
            if dtype not in _PTX_TYPE:
                raise ValueError(f"WGMMA does not support element type {dtype!r}")
        if self.acc_dtype not in ("float32", "float16"):
            raise ValueError(
                f"WGMMA accumulator must be float32 or float16, got {self.acc_dtype!r}"
            )
        if (self.b_type != self.ab_dtype) and not self.is_fp8:
            raise ValueError("Only fp8 WGMMA may mix A and B element types")

    @property
    def b_type(self) -> str:
        return self.b_dtype or self.ab_dtype

    @property
    def is_fp8(self) -> bool:
        return _ELEM_BYTES[self.ab_dtype] == 1

    @property
    def k(self) -> int:
        """K of one instruction: 16 for f16/bf16, 32 for fp8."""
        return _K_BYTES // _ELEM_BYTES[self.ab_dtype]

    @property
    def k_bytes(self) -> int:
        """Bytes of A (and B) along K consumed by one instruction (always 32)."""
        return _K_BYTES

    @property
    def num_acc(self) -> int:
        """Accumulator registers per thread: N/2 for f32, N/4 for packed f16."""
        return self.n // 2 if self.acc_dtype == "float32" else self.n // 4

    @property
    def acc_register_dtype(self) -> str:
        return "float32" if self.acc_dtype == "float32" else "uint32"

    @property
    def num_a_registers(self) -> int:
        """32-bit A-fragment registers per thread in the ``rs`` form (always 4)."""
        return 4

    @property
    def mnemonic(self) -> str:
        d = "f32" if self.acc_dtype == "float32" else "f16"
        return (
            f"wgmma.mma_async.sync.aligned.m64n{self.n}k{self.k}.{d}"
            f".{_PTX_TYPE[self.ab_dtype]}.{_PTX_TYPE[self.b_type]}"
        )

    def mma(self, acc, a, desc_b, scale_d=True, *, transpose_a=False, transpose_b=False):
        """Issue one ``wgmma.mma_async``: ``acc = A @ B (+ acc if scale_d)``.

        ``acc`` is a local tensor of :attr:`num_acc` registers. ``a`` is a
        descriptor (``ss``) or a sequence of :attr:`num_a_registers` uint32
        registers (``rs``). ``scale_d`` False overwrites ``acc``. ``transpose_*``
        selects MN-major operands, which only f16/bf16 support.
        """
        if (transpose_a or transpose_b) and self.is_fp8:
            raise ValueError("fp8 WGMMA operands must be K-major (no transpose)")
        if transpose_a and self.a_in_registers:
            raise ValueError("The rs form has no A transpose")
        d = [acc[i] for i in range(self.num_acc)]
        a_args = list(a) if self.a_in_registers else [a]
        if self.a_in_registers and len(a_args) != self.num_a_registers:
            raise ValueError(f"rs WGMMA takes {self.num_a_registers} A registers")
        tail = [scale_d, 1, 1]
        if not self.is_fp8:
            tail += [] if self.a_in_registers else [int(transpose_a)]
            tail += [int(transpose_b)]
        T.ptx[self.mnemonic](*d, *a_args, desc_b, *tail)


def smem_descriptor(ptr, *, major="K", swizzle=128, mn_atom_stride_bytes=None):
    """A WGMMA shared-memory matrix descriptor for a canonical swizzled tile.

    ``ptr`` is the tile's first element in shared memory, aligned to the swizzle
    atom (1024 B for 128B swizzle).

    * ``major="K"``: rows are M (or N), each row holds ``swizzle`` bytes of K, and
      8-row groups are ``8 * swizzle`` bytes apart (stride byte offset). This is
      what a TMA box of ``(rows, swizzle bytes)`` with the same swizzle writes.
    * ``major="MN"`` (f16/bf16 only; pass ``transpose_*=True`` to :meth:`WGMMA.mma`):
      rows are K, each row holds ``swizzle`` bytes of M (or N), 8-row K groups are
      ``8 * swizzle`` bytes apart, and consecutive ``swizzle``-byte column blocks
      of M (N) are ``mn_atom_stride_bytes`` apart (leading byte offset). A tile
      loaded as one TMA box per column block, stacked, has
      ``mn_atom_stride_bytes = k_rows * swizzle``.

    Returns a ``uint64`` value. Advance it along K with :func:`advance_descriptor`.
    """
    if swizzle not in (32, 64, 128):
        raise ValueError(f"WGMMA descriptors here support 32/64/128B swizzle, got {swizzle}")
    stride = 8 * swizzle
    if major == "K":
        leading = 16  # unused for swizzled K-major layouts; the ISA wants a non-zero value
    elif major == "MN":
        if mn_atom_stride_bytes is None:
            raise ValueError("MN-major descriptors need mn_atom_stride_bytes")
        leading = mn_atom_stride_bytes
    else:
        raise ValueError(f"major must be 'K' or 'MN', got {major!r}")
    for name, value in (("leading", leading), ("stride", stride)):
        if value % 16 or not 0 < value < (1 << 18):
            raise ValueError(
                f"descriptor {name} byte offset {value} must be a multiple of 16 below 256 KiB"
            )
    desc = T.bind_(T.alloc_local([1], "uint64"))
    T.cuda.wgmma.encode_matrix_descriptor(
        T.address_of(desc[0]), ptr, leading >> 4, stride >> 4, _SWIZZLE_CODE[swizzle]
    )
    return desc[0]


def advance_descriptor(desc, byte_offset):
    """``desc`` moved by ``byte_offset`` bytes (a multiple of 16) in shared memory.

    For a K-major tile, K step ``k`` of an instruction is
    ``advance_descriptor(desc, k * wgmma.k_bytes)``; for an MN-major tile it is
    ``k * wgmma.k * swizzle`` (``wgmma.k`` rows of ``swizzle`` bytes).
    """
    if isinstance(byte_offset, int) and byte_offset % 16:
        raise ValueError(f"descriptor offsets must be multiples of 16 bytes, got {byte_offset}")
    return (
        desc + T.uint64(byte_offset >> 4)
        if isinstance(byte_offset, int)
        else desc + (T.Cast("uint64", byte_offset) >> T.uint64(4))
    )


def fence_operands(*tensors):
    """Pin register operands across the async MMA (CUTLASS ``warpgroup_fence_operand``).

    Call on the accumulator (and rs A registers) before ``wgmma.fence`` and after
    ``wgmma.wait_group``, so nvcc neither moves accesses across the MMA nor copies
    the registers.
    """
    for tensor, count in tensors:
        for i in range(count):
            T.cuda.wgmma.noop_barrier(tensor[i])


def chain(
    op: WGMMA, acc, a_steps, b_steps, *, accumulate=True, transpose_a=False, transpose_b=False
):
    """One WGMMA chain: fence, ``len(b_steps)`` back-to-back MMAs, commit.

    ``a_steps`` / ``b_steps`` give the A operand (descriptor, or a sequence of 4
    registers for rs) and the B descriptor of each k step. ``accumulate`` is the
    ``scale_d`` of the first step (False starts a new tile); later steps always
    accumulate. Nothing else may touch ``acc`` until :func:`wait` returns, or
    ptxas serializes the chain.
    """
    if len(a_steps) != len(b_steps) or not b_steps:
        raise ValueError("chain needs one A and one B operand per k step")
    regs = [(acc, op.num_acc)]
    if op.a_in_registers:
        regs += [(a, op.num_a_registers) for a in a_steps]
    fence_operands(*regs)
    T.ptx.wgmma.fence.sync.aligned()
    for k, (a, b) in enumerate(zip(a_steps, b_steps)):
        op.mma(
            acc,
            a,
            b,
            accumulate if k == 0 else True,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )
    T.ptx.wgmma.commit_group.sync.aligned()


def wait(pending, *fence):
    """``wgmma.wait_group pending``, then fence the given ``(tensor, count)`` operands."""
    T.ptx.wgmma.wait_group.sync.aligned(pending)
    fence_operands(*fence)


def acc_coord(thread, reg):
    """(row, col) in the 64 x N tile of accumulator register ``reg`` of warpgroup
    thread ``thread`` (0..127), for f32 accumulators (ISA 9.7.17.5.1.1.1).

    ``row = 16 * warp + lane // 4 + 8 * ((reg % 4) // 2)``,
    ``col = 8 * (reg // 4) + 2 * (lane % 4) + reg % 2``. ``reg`` may be a Python
    int (typical, in an unrolled loop) or an expression.
    """
    warp, lane = thread // 32, thread % 32
    row = 16 * warp + lane // 4 + 8 * ((reg % 4) // 2)
    col = 8 * (reg // 4) + 2 * (lane % 4) + reg % 2
    return row, col


def a_fragment_coord(thread, reg, half):
    """(row, k) in the 64 x 16 (f16/bf16) A tile held by half ``half`` (0, 1) of rs
    A register ``reg`` (0..3) of warpgroup thread ``thread`` (ISA 9.7.17.5.1.1.3).

    The layout equals :func:`acc_coord` for accumulator registers
    ``2 * reg + half`` of a 16-column slice, so packing accumulator pairs
    ``(2j, 2j + 1)`` of columns ``16s .. 16s + 15`` yields the A fragment of k
    slice ``s``.
    """
    return acc_coord(thread, 2 * reg + half)
