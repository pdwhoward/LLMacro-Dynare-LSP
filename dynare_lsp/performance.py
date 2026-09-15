"""Deterministic work-unit profiling for the Dynare analysis pipeline.

Wall-clock timings are noisy across machines and Python versions.  This module
instead records stable units of work -- parser invocations, include-resolution
attempts, workspace walks, and similar counters -- only while an explicit
profile capture is active.  The normal editor hot path pays one cheap
``ContextVar.get()`` per instrumented operation and allocates nothing.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


@dataclass
class WorkProfile:
    """Mutable counter set collected by :func:`capture_work`."""

    counters: Dict[str, int] = field(default_factory=dict)

    def record(self, name: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        self.counters[name] = self.counters.get(name, 0) + int(amount)

    def count(self, name: str) -> int:
        return self.counters.get(name, 0)

    def to_dict(self) -> Dict[str, int]:
        return dict(sorted(self.counters.items()))


_ACTIVE_PROFILE: ContextVar[Optional[WorkProfile]] = ContextVar(
    "dynare_lsp_work_profile",
    default=None,
)


def record_work(name: str, amount: int = 1) -> None:
    """Add deterministic work to the active profile, if any."""
    profile = _ACTIVE_PROFILE.get()
    if profile is not None:
        profile.record(name, amount)


@contextmanager
def capture_work() -> Iterator[WorkProfile]:
    """Capture work counters for the current context.

    Captures are context-local, so concurrent LSP requests do not contaminate
    one another.  Nested captures intentionally isolate their counters.
    """
    profile = WorkProfile()
    token = _ACTIVE_PROFILE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE_PROFILE.reset(token)
