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
"""Multicast consumer release (MBarrier.arrive_multicast) and max_active_clusters."""

import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.lang import SMEMPool
from tvm_tirx.backend.cuda.lang.pipeline import MBarrier
from tvm_tirx.backend.cuda.runtime import occupancy
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

requires_clusters = pytest.mark.skipif(
    not env.has_cuda_compute(9), reason="thread-block clusters need compute >= 9.0"
)


def _compile(func):
    return tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target="cuda", pipeline="tirx")


@requires_clusters
@pytest.mark.parametrize("cluster", [2, 4])
def test_arrive_multicast_releases_every_cta(cluster):
    """Each CTA's elected thread arrives on slot 0 in every CTA of the cluster.

    A barrier initialized with count == cluster completes only when all CTAs'
    arrivals land, so every CTA's wait returns and it records how many rounds it
    saw. Three rounds also exercise the phase flip.
    """
    import torch

    rounds = 3

    # fmt: off
    @T.function
    def main(done: T.Tensor((cluster,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=cluster, block=128, cluster=cluster))
        tx = T.cuda.thread_idx("x")
        rank = T.cuda.mov_sreg(32, "cluster_ctarank")
        pool = SMEMPool()
        empty = MBarrier(pool, 1)
        pool.commit()
        empty.init(cluster)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cluster_sync()
        for r in T.serial(rounds):
            if tx == 0:
                empty.arrive_multicast(0, ranks=range(cluster))
            empty.wait(0, r % 2)
        if tx == 0:
            done[rank] = rounds
        T.cuda.cluster_sync()
    # fmt: on

    done = torch.zeros(cluster, dtype=torch.int32, device="cuda")
    _compile(main)(done)
    torch.cuda.synchronize()
    assert done.tolist() == [rounds] * cluster


_TMA_MC = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
)


@requires_clusters
@pytest.mark.parametrize("stages", [2, 3])
@pytest.mark.parametrize("skew", [False, True])
def test_multicast_ring_under_phase_wrap(stages, skew):
    """A TMA-multicast ring with multicast release, wrapping each stage's phase hundreds of times.

    2-CTA cluster. Per iteration, each CTA's producer multicasts one 1 KB row into the
    stage of *both* CTAs; 128 consumers per CTA sum both rows, then release the stage in
    both CTAs with ``empty.arrive_multicast(stage, ranks=[0, 1])`` (empty count 2). A
    producer can only overwrite a stage once both CTAs' consumers have released it, so
    the sums are exact. ``skew`` makes CTA 1's consumers sleep every 5th iteration, so
    the CTAs drift apart and the cross-CTA release is what holds them together.
    (Reduced from the review's stress kernel: 4096 iterations x 3 launches, all exact.)
    """
    import torch

    nk, width = 512, 256

    # fmt: off
    @T.function
    def main(X: T.Tensor((nk * 2, width), "float32", align=16),
             out: T.Tensor((2, 128, 4), "float32")):
        X_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", X_map, "float32", 2, X.data, width, nk * 2,
                      width * 4, width, 1, 1, 1, 0, 0, 0, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=256, cluster=2))
        tx = T.cuda.thread_idx("x")
        rank = T.cuda.mov_sreg(32, "cluster_ctarank")
        pool = SMEMPool()
        full = MBarrier(pool, stages)
        empty = MBarrier(pool, stages)
        pool.commit()
        buf = T.alloc_shared((stages * 2 * width,), "float32", align=1024)
        acc = T.alloc_local((4,), "float32")
        if tx == 0:
            full.init(1)
            empty.init(2)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cluster_sync()
        if tx == 0:
            for kt in T.serial(nk):
                stage = T.meta_var(kt % stages)
                empty.wait(stage, ((kt // stages) & 1) ^ 1)
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(full.ptr_to([stage]),
                                                           T.uint32(2 * width * 4))
                T.ptx[_TMA_MC](buf.ptr_to([stage * 2 * width + rank * width]),
                               T.address_of(X_map), 0, kt * 2 + rank, full.ptr_to([stage]),
                               T.uint16(3))
        elif tx >= 128:
            c = tx - 128
            for i in T.unroll(4):
                acc[i] = T.float32(0)
            for kt in T.serial(nk):
                stage = T.meta_var(kt % stages)
                full.wait(stage, (kt // stages) & 1)
                if T.constexpr(skew):
                    if rank == 1:
                        if kt % 5 == 0:
                            T.cuda.nano_sleep(T.uint64(3000))
                for i in T.unroll(4):
                    row = stage * 2 * width + (i // 2) * width
                    acc[i] = acc[i] + buf[row + c * 2 + (i % 2)]
                T.ptx["bar.sync"](T.uint32(1), T.uint32(128))
                if c == 0:
                    empty.arrive_multicast(stage, ranks=[0, 1])
            for i in T.unroll(4):
                out[rank, c, i] = acc[i]
        T.cuda.cluster_sync()
    # fmt: on

    mod = _compile(main)
    torch.manual_seed(0)
    X = torch.randint(-8, 8, (nk * 2, width), device="cuda").float()  # sums stay exact in fp32
    out = torch.zeros(2, 128, 4, device="cuda")
    rows = X.view(nk, 2, width)
    ref = torch.stack(
        [torch.stack([rows[:, j // 2, c * 2 + j % 2].sum() for j in range(4)]) for c in range(128)]
    )
    for _ in range(2):
        out.zero_()
        mod(X, out)
        torch.cuda.synchronize()
        assert torch.equal(out[0], ref)
        assert torch.equal(out[1], ref)


def test_arrive_multicast_rejects_empty_ranks():
    with pytest.raises(ValueError, match="at least one"):

        @T.function
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
            pool = SMEMPool()
            bar = MBarrier(pool, 1)
            pool.commit()
            bar.arrive_multicast(0, ranks=[])


def _raw_max_active_clusters(binary, name, block, cluster):
    from cuda.bindings import driver as d

    def ck(result):
        assert result[0] == d.CUresult.CUDA_SUCCESS, result
        return result[1] if len(result) == 2 else None

    ck(d.cuInit(0))
    dev = ck(d.cuDeviceGet(0))
    ctx = ck(d.cuDevicePrimaryCtxRetain(dev))
    ck(d.cuCtxPushCurrent(ctx))
    try:
        mod = ck(d.cuModuleLoadData(binary))
        fn = ck(d.cuModuleGetFunction(mod, name.encode()))
        if cluster > 8:
            attr = d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
            ck(d.cuFuncSetAttribute(fn, attr, 1))
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 132 * cluster, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = 0
        a = d.CUlaunchAttribute()
        a.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        a.value.clusterDim.x, a.value.clusterDim.y, a.value.clusterDim.z = cluster, 1, 1
        cfg.attrs = [a]
        cfg.numAttrs = 1
        n = ck(d.cuOccupancyMaxActiveClusters(fn, cfg))
        d.cuModuleUnload(mod)
        return n
    finally:
        d.cuCtxPopCurrent()
        d.cuDevicePrimaryCtxRelease(dev)


def _persistent_like(block, smem_elems, cluster):
    # fmt: off
    @T.function
    def main(out: T.Tensor((1,), "float32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=cluster, block=block, cluster=cluster),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1),
        )
        tx = T.cuda.thread_idx("x")
        buf = T.alloc_shared((smem_elems,), "float32", align=1024)
        buf[tx] = T.Cast("float32", tx)
        T.cuda.cta_sync()
        if tx == 0:
            out[0] = buf[block - 1]
    # fmt: on
    return main


@requires_clusters
@pytest.mark.parametrize("block, smem_kb", [(384, 192), (256, 96), (128, 32)])
@pytest.mark.parametrize("cluster", [1, 2, 4, 8])
def test_max_active_clusters_matches_the_driver(block, smem_kb, cluster):
    pytest.importorskip("cuda.bindings")
    exe = _compile(_persistent_like(block, smem_kb * 1024 // 4, cluster))
    binary, name = occupancy.kernel_binary(exe)
    expected = _raw_max_active_clusters(binary, name, block, cluster)
    assert occupancy.max_active_clusters(exe, block=block, cluster=cluster) == expected
    assert expected >= 1


@pytest.mark.skipif(
    not (env.has_cuda_compute(9, exact=True) and env.has_cuda()),
    reason="H200/H100 regression values",
)
def test_max_active_clusters_is_not_num_sms_over_cluster_on_hopper():
    """The plan's probe-GEMM configuration (384 threads, ~192 KB smem): cluster 4
    fits fewer clusters than num_sms // 4 on a 132-SM Hopper part."""
    pytest.importorskip("cuda.bindings")
    import torch

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    got = {
        c: occupancy.max_active_clusters(
            _compile(_persistent_like(384, 192 * 1024 // 4, c)), block=384, cluster=c
        )
        for c in (1, 2, 4)
    }
    assert got[1] == num_sms
    if num_sms == 132:
        assert got[4] < num_sms // 4
