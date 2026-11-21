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
"""The NVRTC and CUDA-header versions the production compile path uses (for ``floors``)."""

from __future__ import annotations

import os
import re


def _cudart_header_version() -> int | None:
    """``CUDART_VERSION`` of the ``cuda_runtime_api.h`` the production NVRTC path includes."""
    from tvm_tirx.support import nvcc  # pylint: disable=import-outside-toplevel

    cuda_path = nvcc.find_cuda_path()
    target_include = nvcc._find_cuda_target_include(cuda_path)  # pylint: disable=protected-access
    for include in (os.path.join(cuda_path, "include"), target_include):
        header = os.path.join(include, "cuda_runtime_api.h") if include else None
        if header and os.path.isfile(header):
            with open(header, encoding="utf-8", errors="replace") as f:
                match = re.search(r"^#define\s+CUDART_VERSION\s+(\d+)", f.read(), re.M)
            if match:
                return int(match.group(1))
    return None


def current_toolchain() -> dict | None:
    """``{"nvrtc": (major, minor), "cudart_headers": int}`` of this process, or None if unknown."""
    try:
        from cuda.bindings import nvrtc  # pylint: disable=import-outside-toplevel

        status, major, minor = nvrtc.nvrtcVersion()
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return None
        return {"nvrtc": (int(major), int(minor)), "cudart_headers": _cudart_header_version()}
    except Exception:
        return None
