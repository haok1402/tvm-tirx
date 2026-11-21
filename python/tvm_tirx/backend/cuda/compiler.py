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
"""Compile one CUDA source group using explicit backend settings."""

from typing import NamedTuple

from ..config import copy_backend_config
from .backend_config import compiler_options, resolve_backend_config
from .perf_remarks import PerfRemark, parse_compiler_log, report_perf_remarks


class CompilationResult(NamedTuple):
    """Device artifact, the effective settings used to produce it, and the compile log.

    ``remarks`` are the performance remarks parsed from ``log`` (see
    :mod:`tvm_tirx.backend.cuda.perf_remarks`).
    """

    binary: bytes
    target_format: str
    config: dict
    log: str = ""
    remarks: tuple[PerfRemark, ...] = ()


def compile_source(source: str, backend_config=None, *, target=None) -> CompilationResult:
    """Compile source with per-backend overrides and optional Target/tag defaults.

    When target is omitted, use the configuration architecture or detect the device.
    Compiler logs are reported on failure. On success the log is returned on the
    result, and its performance remarks (WGMMA serialization, spills, stack frames)
    are handled by ``backend_config.perf_remarks``: "warn" (default), "error" or "ignore".
    """
    from tvm_tirx.support import nvcc

    config = resolve_backend_config(copy_backend_config(backend_config).get("cuda"), target)
    use_nvshmem = "#include <nvshmem.h>" in source or "#include <nvshmemx.h>" in source
    output_format = config.get("target_format") or (
        "cubin" if config["compiler"] == "nvrtc" or use_nvshmem else "fatbin"
    )
    if use_nvshmem and output_format != "cubin":
        raise ValueError("NVSHMEM device linking requires target_format='cubin'")
    config["target_format"] = output_format
    driver = getattr(nvcc, f"_compile_cuda_{config['compiler']}")
    binary, log = driver(
        source, output_format, config["arch"], compiler_options(config), use_nvshmem=use_nvshmem
    )
    log = log or ""
    remarks = parse_compiler_log(log)
    report_perf_remarks(remarks, config.get("perf_remarks", "warn"), arch=config["arch"])
    return CompilationResult(bytes(binary), output_format, {"cuda": config}, log, remarks)
