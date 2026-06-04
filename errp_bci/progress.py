"""Kaggle-friendly progress reporting.

``tqdm.auto`` renders as a Jupyter *widget* that does NOT appear in Kaggle's
committed run logs — so long training loops looked frozen ("Supervised done"
with nothing in between). This module provides a drop-in ``tqdm`` replacement
that instead prints plain, flushed, periodic text lines to stdout, which always
show up in the Kaggle log:

    [Full-MAML S05 meta-train] 500/2000 (25%)   12s  eta   36s

It updates at most once every ``every_sec`` seconds (and/or every ``every_n``
iterations), so it never floods the log. API is a subset of tqdm's, sufficient
for how the pipeline uses it: ``tqdm(iterable, desc=..., total=..., every_n=...,
leave=...)``.
"""
import sys
import time


class _Progress:
    def __init__(self, iterable=None, total=None, desc="", every_sec=15.0,
                 every_n=None, leave=True, **_ignored):
        self.iterable = iterable if iterable is not None else range(total or 0)
        if total is not None:
            self.total = total
        elif hasattr(self.iterable, "__len__"):
            self.total = len(self.iterable)
        else:
            self.total = None
        self.desc = desc
        self.every_sec = every_sec
        self.every_n = every_n
        self.leave = leave
        self.n = 0
        self._start = time.time()
        self._last = 0.0

    def __iter__(self):
        self._start = time.time()
        self._emit(force=True)
        for obj in self.iterable:
            yield obj
            self.n += 1
            self._maybe_emit()
        self._emit(force=True, done=True)

    def _maybe_emit(self):
        now = time.time()
        hit_n = self.every_n is not None and (self.n % self.every_n == 0)
        hit_t = (now - self._last) >= self.every_sec
        if hit_n or hit_t:
            self._emit()

    def _emit(self, force=False, done=False):
        if not force and not done and self.n == 0:
            return
        el = time.time() - self._start
        if self.total:
            frac = f"{self.n}/{self.total}"
            pct = f" ({100.0 * self.n / self.total:3.0f}%)"
        else:
            frac, pct = f"{self.n}", ""
        eta = ""
        if self.total and self.n > 0 and not done:
            rate = self.n / el if el > 0 else 0.0
            if rate > 0:
                eta = f"  eta {(self.total - self.n) / rate:5.0f}s"
        tag = f"[{self.desc}] " if self.desc else ""
        suffix = "  done" if done else eta
        print(f"  {tag}{frac}{pct}  {el:5.0f}s{suffix}", flush=True)
        self._last = time.time()

    # minimal tqdm-compatible no-ops in case anything calls them
    def update(self, k=1):
        self.n += k
        self._maybe_emit()

    def set_description(self, desc):
        self.desc = desc

    @staticmethod
    def write(msg, *_a, **_k):
        print(msg, flush=True)


# Drop-in name used by the rest of the package.
tqdm = _Progress
