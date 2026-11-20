# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run the Petit distributed reference test for Kimi K3 with 448 experts (one node's
half in the two-level EP16 layout). Launch: torchrun --nproc-per-node 8 k3half_test.py
"""

import dataclasses
import sys

import tokenspeed_kernel.ops.moe.gluon as g

half = dataclasses.replace(
    g._KIMI_K3_PROFILE, num_experts=448, world_sizes=frozenset({8})
)
g._PROFILES = tuple(half if p is g._KIMI_K3_PROFILE else p for p in g._PROFILES)
g._KIMI_K3_PROFILE = half
sys.path.insert(0, sys.argv[1])  # test directory
sys.path.insert(0, sys.argv[1].split("/test/")[0] + "/test")  # test root: utils.py
import test_gluon_petit_distributed as T

T._PROFILES["kimi_k3"] = dataclasses.replace(T._PROFILES["kimi_k3"], experts=448)
T.test_distributed_petit_reference("kimi_k3")
import torch.distributed as dist

print(f"rank {dist.get_rank()} PASS", flush=True)
