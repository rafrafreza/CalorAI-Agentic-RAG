"""
Lightweight in-process latency telemetry.

Rolling windows (last N samples) per metric with nearest-rank percentiles. Every
chat turn records its end-to-end latency (split by text vs image path) and every
graph node records its own span, so ``GET /metrics`` answers "where does the time go?".
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


class LatencyRecorder:
    def __init__(self, window: int = 1000) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._lock = threading.Lock()

    def record(self, metric: str, ms: float) -> None:
        with self._lock:
            self._samples[metric].append(ms)

    def stats(self, metric: str) -> dict:
        with self._lock:
            values = sorted(self._samples.get(metric, ()))
        if not values:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(values),
            "p50_ms": round(percentile(values, 50), 1),
            "p95_ms": round(percentile(values, 95), 1),
            "p99_ms": round(percentile(values, 99), 1),
            "mean_ms": round(statistics.fmean(values), 1),
            "max_ms": round(values[-1], 1),
        }

    def snapshot(self) -> dict:
        with self._lock:
            names = sorted(self._samples)
        grouped: dict[str, dict] = {}
        for name in names:
            group, _, key = name.partition(".")
            grouped.setdefault(group, {})[key or group] = self.stats(name)
        return grouped

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


recorder = LatencyRecorder()


@contextmanager
def span(name: str, sink: list[dict] | None = None, **attrs):
    """Time a block; record it under ``node.<name>`` and optionally append a trace span."""
    start = time.perf_counter()
    info: dict = {"name": name, **attrs}
    try:
        yield info
    finally:
        ms = (time.perf_counter() - start) * 1000
        info["ms"] = round(ms, 1)
        recorder.record(f"node.{name}", ms)
        if sink is not None:
            sink.append(info)
