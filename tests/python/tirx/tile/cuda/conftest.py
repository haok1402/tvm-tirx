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
"""Hardware requirements for CUDA tile-op tests."""

from pathlib import Path

from tvm_tirx.testing._pytest import cuda_compute_collection_modifyitems


def pytest_collection_modifyitems(items):
    # Every test here, GPU or source-only, needs a device of at least its
    # cuda_min_compute (default 10.0).
    cuda_compute_collection_modifyitems(items, Path(__file__).resolve().parent, gpu_only=False)
