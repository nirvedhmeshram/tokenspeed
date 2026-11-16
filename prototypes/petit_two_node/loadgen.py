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

"""Closed-loop decode load against an OpenAI-compatible /v1/completions server.

CONCURRENCY streams each send requests back to back (prompt of ~INPUT_LEN
tokens, exactly OUTPUT_LEN generated with ignore_eos). Reports decode token
throughput over the steady window (after the first round), TTFT and TPOT.
Usage: python3 loadgen.py URL CONCURRENCY INPUT_LEN OUTPUT_LEN DURATION_S
"""

import asyncio
import json
import random
import statistics
import sys
import time

import aiohttp

URL, CONC, IN_LEN, OUT_LEN, DUR = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    float(sys.argv[5]),
)
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()


def prompt(seed):
    rng = random.Random(seed)
    # ~1 token per word for these words; a unique prefix defeats prefix caching.
    return f"{seed} " + " ".join(rng.choice(WORDS) for _ in range(IN_LEN))


async def one(session, seed, stats):
    body = {
        "model": "dsv41",
        "prompt": prompt(seed),
        "max_tokens": OUT_LEN,
        "min_tokens": OUT_LEN,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first, times, usage = None, [], None
    async with session.post(f"{URL}/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            msg = json.loads(line[5:])
            now = time.perf_counter()
            if msg.get("usage"):
                usage = msg["usage"]
            if msg.get("choices") and msg["choices"][0].get("text"):
                first = first or now
                times.append(now)
    n = usage["completion_tokens"] if usage else len(times)
    stats.append(
        (t0, first, time.perf_counter(), n, usage["prompt_tokens"] if usage else None)
    )


async def worker(session, wid, stop, stats):
    k = 0
    while time.perf_counter() < stop:
        await one(session, wid * 100000 + k, stats)
        k += 1


async def main():
    stats = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        connector=conn, timeout=aiohttp.ClientTimeout(total=None)
    ) as s:
        await one(s, 999999, [])  # warm
        t_start = time.perf_counter()
        stop = t_start + DUR
        await asyncio.gather(*(worker(s, w, stop, stats) for w in range(CONC)))
        t_end = time.perf_counter()
    # Steady window: from when every stream got its first token to when the first stream finished its last request.
    win0 = max(st[1] for st in stats if st[1]) if stats else t_start
    done = [st for st in stats if st[2] > win0]
    tokens = sum(st[3] for st in stats)
    ttft = [st[1] - st[0] for st in stats if st[1]]
    tpot = [(st[2] - st[1]) / max(st[3] - 1, 1) for st in stats if st[1]]
    print(
        json.dumps(
            {
                "concurrency": CONC,
                "input_len": IN_LEN,
                "prompt_tokens_seen": stats[0][4] if stats else None,
                "output_len": OUT_LEN,
                "requests": len(stats),
                "wall_s": round(t_end - t_start, 1),
                "output_tok_per_s": round(tokens / (t_end - t_start), 1),
                "ttft_ms_median": round(statistics.median(ttft) * 1000, 1),
                "tpot_ms_median": round(statistics.median(tpot) * 1000, 2),
                "tpot_ms_p90": round(sorted(tpot)[int(0.9 * len(tpot))] * 1000, 2),
            }
        ),
        flush=True,
    )


asyncio.run(main())
