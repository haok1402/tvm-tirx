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
"""Occupancy queries for compiled CUDA kernels.

Persistent and cluster kernels must size their grid to what the device can keep
resident. For clusters that is **not** ``num_sms // cluster_size``: cluster
placement is constrained per GPC. On an H200 (132 SMs), a 384-thread kernel with
about 192 KB of shared memory fits 30 clusters of 4 (120 CTAs), not 33, and 15
clusters of 8, not 16. A grid of ``num_sms // 4`` clusters would put three
clusters in a second wave, roughly doubling the tail.
:func:`max_active_clusters` asks the driver (``cuOccupancyMaxActiveClusters``).

A persistent kernel whose grid is fixed at trace time can be built in two passes.
Occupancy does not depend on the grid size, so compile once with any grid, query,
and rebuild with ``grid = max_active_clusters(...) * cluster_size``. The helper
needs the ``cuda-bindings`` package (``cuda`` extra).
"""

from __future__ import annotations

import math


def _driver():
    try:
        from cuda.bindings import driver  # pylint: disable=import-outside-toplevel
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("max_active_clusters needs the cuda-bindings package") from err
    return driver


def _check(driver, result):
    status, *values = result if isinstance(result, tuple) else (result,)
    if status != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(status)
        raise RuntimeError(f"CUDA driver call failed: {name.decode() if name else status}")
    return values[0] if len(values) == 1 else values


def _device_binaries(module):
    """Yield (binary, format, kernel names) of every CUDA device module under ``module``."""
    seen = set()

    def walk(mod):
        if id(mod) in seen:
            return
        seen.add(id(mod))
        if mod.kind == "cuda":
            binary, fmt, names = mod["__tvm_cuda_binary"]()
            yield bytes(binary), str(fmt), tuple(str(n) for n in names)
        for imported in mod.imports:
            yield from walk(imported)

    yield from walk(module)


def kernel_binary(executable, kernel=None):
    """The compiled binary and kernel name of ``kernel`` (default: the only kernel)."""
    module = getattr(executable, "mod", executable)
    candidates = []
    for binary, fmt, names in _device_binaries(module):
        if fmt == "cuda":
            raise ValueError(
                "the module holds CUDA source (fallback build); compile it for a device first"
            )
        for name in names:
            if kernel is None or name == kernel:
                candidates.append((binary, name))
    if not candidates:
        raise ValueError(f"no CUDA kernel {kernel!r} in the module")
    if len(candidates) > 1 and kernel is None:
        raise ValueError(f"several CUDA kernels; pass kernel= one of {[n for _, n in candidates]}")
    return candidates[0]


def max_active_clusters(
    executable,
    kernel=None,
    *,
    block,
    cluster,
    dynamic_shared_memory_bytes=0,
    device_id=0,
):
    """Maximum number of clusters of ``kernel`` that can be resident at once.

    Parameters
    ----------
    executable : tvm_tirx.codegen.Executable or runtime Module
        The result of :func:`tvm_tirx.compile`.
    kernel : str, optional
        Kernel symbol (for example ``"main_kernel"``). Optional when the module
        has one kernel.
    block : int or tuple of int
        Threads per CTA, as launched.
    cluster : int or tuple of int
        Cluster shape, as launched.
    dynamic_shared_memory_bytes : int
        Dynamic shared memory per CTA. TIRx ``alloc_shared`` buffers are static
        and already part of the binary; pass this only for ``shared.dyn``.
    device_id : int
        CUDA device ordinal.
    """
    driver = _driver()
    block = tuple(block) if isinstance(block, (tuple, list)) else (block,)
    cluster = tuple(cluster) if isinstance(cluster, (tuple, list)) else (cluster,)
    block = block + (1,) * (3 - len(block))
    cluster = cluster + (1,) * (3 - len(cluster))
    binary, name = kernel_binary(executable, kernel)

    _check(driver, driver.cuInit(0))
    device = _check(driver, driver.cuDeviceGet(device_id))
    context = _check(driver, driver.cuDevicePrimaryCtxRetain(device))
    try:
        _check(driver, driver.cuCtxPushCurrent(context))
        try:
            cu_module = _check(driver, driver.cuModuleLoadData(binary))
            try:
                function = _check(driver, driver.cuModuleGetFunction(cu_module, name.encode()))
                if math.prod(cluster) > 8:
                    attrs = driver.CUfunction_attribute
                    attr = attrs.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
                    _check(driver, driver.cuFuncSetAttribute(function, attr, 1))
                if dynamic_shared_memory_bytes:
                    attr = (
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                    )
                    _check(
                        driver,
                        driver.cuFuncSetAttribute(function, attr, dynamic_shared_memory_bytes),
                    )
                config = driver.CUlaunchConfig()
                # The query ignores the grid beyond requiring a multiple of the cluster shape.
                config.gridDimX, config.gridDimY, config.gridDimZ = cluster
                config.blockDimX, config.blockDimY, config.blockDimZ = block
                config.sharedMemBytes = dynamic_shared_memory_bytes
                attribute = driver.CUlaunchAttribute()
                attribute.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                (
                    attribute.value.clusterDim.x,
                    attribute.value.clusterDim.y,
                    attribute.value.clusterDim.z,
                ) = cluster
                config.attrs = [attribute]
                config.numAttrs = 1
                return int(_check(driver, driver.cuOccupancyMaxActiveClusters(function, config)))
            finally:
                driver.cuModuleUnload(cu_module)
        finally:
            driver.cuCtxPopCurrent()
    finally:
        driver.cuDevicePrimaryCtxRelease(device)
