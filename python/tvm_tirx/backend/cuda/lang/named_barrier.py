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
consumer, and so on. Barrier 0 is what ``T.cuda.cta_sync()`` (``__syncthreads``)
uses, so it is reserved. :class:`NamedBarriers` hands out ids 1..15 at trace time
so independent helpers in one kernel never collide.
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
    """Trace-time allocator of barrier ids 1..15 for one kernel."""

    def __init__(self, first_id: int = 1):
        if first_id in RESERVED_BARRIER_IDS or first_id < 0:
            raise ValueError(f"first_id {first_id} is reserved")
        self._next = first_id

    def alloc(self, num_threads: int) -> NamedBarrier:
        if self._next >= NUM_HARDWARE_BARRIERS:
            raise ValueError(f"out of named barriers (ids 1..{NUM_HARDWARE_BARRIERS - 1})")
        barrier = NamedBarrier(self._next, num_threads)
        self._next += 1
        return barrier
