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
"""Named CTA barriers (``bar.sync`` / ``bar.arrive`` with an id and a thread count).

Warp-specialized kernels order warpgroups with named barriers: two consumer
warpgroups ping-pong their MMA and epilogue phases, a producer signals a
consumer, and so on. There are 16 hardware barriers per CTA.

**Ids are a kernel-wide resource.** Create **one** :class:`NamedBarriers` per
kernel, like the kernel's ``SMEMPool``, and pass it to every helper that needs
barriers, so that no two of them get the same id. Barrier 0 is
``T.cuda.cta_sync()`` (``__syncthreads``) and is always reserved.

Code that names ids directly, ``T.cuda.warpgroup_sync(id)`` (``bar.sync id, 128``)
or raw ``T.ptx["bar.sync"]``, must :meth:`NamedBarriers.reserve` those ids in the
kernel's allocator. Known users:

* ``ClusterLaunchControlScheduler`` (sm_100 CLC worker,
  ``lang/tile_scheduler.py``) uses ``warpgroup_sync(wg_id + 1)``, i.e. ids
  ``1 .. num_warpgroups``;
* TIRx-kernels kernels that pick ids by hand (e.g. ``wg + 10``).
"""

from __future__ import annotations

from tvm_tirx.script import tirx as T

NUM_HARDWARE_BARRIERS = 16
RESERVED_BARRIER_IDS = (0,)  # barrier 0 is __syncthreads / T.cuda.cta_sync()


class NamedBarrier:
    """One hardware barrier shared by ``num_threads`` threads (a multiple of 32).

    ``sync()`` arrives and waits; ``arrive()`` arrives without waiting. A phase
    completes when ``num_threads`` threads have arrived, counting both.
    """

    def __init__(self, barrier_id: int, num_threads: int):
        if barrier_id in RESERVED_BARRIER_IDS:
            raise ValueError(f"named barrier id {barrier_id} is reserved for cta_sync")
        if not 0 <= barrier_id < NUM_HARDWARE_BARRIERS:
            raise ValueError(f"named barrier id must be in [1, {NUM_HARDWARE_BARRIERS - 1}]")
        if num_threads <= 0 or num_threads % 32:
            raise ValueError(
                f"named barrier thread count {num_threads} must be a positive multiple of 32"
            )
        self.barrier_id = barrier_id
        self.num_threads = num_threads

    def sync(self):
        T.ptx["bar.sync"](T.uint32(self.barrier_id), T.uint32(self.num_threads))

    def arrive(self):
        T.ptx["bar.arrive"](T.uint32(self.barrier_id), T.uint32(self.num_threads))

    def __repr__(self):
        return f"NamedBarrier(id={self.barrier_id}, threads={self.num_threads})"


class NamedBarriers:
    """Trace-time allocator of barrier ids 1..15 for **one kernel**.

    ``reserved`` (or later :meth:`reserve` calls) marks ids that code in the same
    kernel names directly, e.g. through ``T.cuda.warpgroup_sync``. :meth:`alloc`
    never returns a reserved id, and :meth:`reserve` refuses an id already
    handed out.
    """

    def __init__(self, reserved=()):
        self._used = set(RESERVED_BARRIER_IDS)
        self._allocated = set()
        self.reserve(*reserved)

    def reserve(self, *ids: int):
        """Mark ``ids`` as taken by code that uses them directly."""
        for barrier_id in ids:
            if not 0 < barrier_id < NUM_HARDWARE_BARRIERS:
                raise ValueError(
                    f"named barrier id {barrier_id} is outside 1..{NUM_HARDWARE_BARRIERS - 1}"
                )
            if barrier_id in self._allocated:
                raise ValueError(
                    f"named barrier id {barrier_id} was already allocated in this kernel"
                )
            self._used.add(barrier_id)
        return self

    def alloc(self, num_threads: int) -> NamedBarrier:
        """A barrier with the lowest free id, for ``num_threads`` threads."""
        for barrier_id in range(1, NUM_HARDWARE_BARRIERS):
            if barrier_id not in self._used:
                self._used.add(barrier_id)
                self._allocated.add(barrier_id)
                return NamedBarrier(barrier_id, num_threads)
        raise ValueError(f"out of named barriers (ids 1..{NUM_HARDWARE_BARRIERS - 1} all taken)")

    @property
    def free(self) -> int:
        """How many ids are still available."""
        return NUM_HARDWARE_BARRIERS - len(self._used)
