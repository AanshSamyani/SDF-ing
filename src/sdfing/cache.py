"""Tiny on-disk cache mapping a training-config key -> tinker:// adapter path.

Lets `run_ip_experiment.py` skip retraining and re-run just the eval. The cache
file lives under the repo (e.g. outputs/adapters.json), which is gitignored and
persists under workspace/. Tinker sampler weights saved with ttl_seconds=None
don't expire, so cached paths stay valid across sessions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class AdapterCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, str] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Merge with whatever is on disk (a parallel run may have added other keys),
        # then write atomically so concurrent processes don't clobber each other.
        on_disk: dict[str, str] = {}
        if self.path.exists():
            try:
                on_disk = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                on_disk = {}
        on_disk.update(self.data)
        self.data = on_disk
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
