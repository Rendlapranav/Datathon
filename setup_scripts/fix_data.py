"""
fix_data.py — DEPRECATED, kept for reference only.

This script called load_dataset(..., streaming=False) on both the full
Electronics reviews AND metadata configs, which downloads and caches the
entire ~40M-row dataset to disk and loads it into memory via
.to_pandas() before doing anything else. That is not safe on a laptop
with limited RAM/disk, and it is not necessary — the McAuley streaming
API supports iterating without ever materializing the full dataset.

setup_scripts/stream_data.py replaces this: it uses streaming=True
throughout (rows pulled one at a time, discarded after use), applies the
same Cochran's-n statistical-significance reasoning this script used, and
additionally selects direct competitor products for benchmarking.

Run instead:
    python setup_scripts/stream_data.py
"""
import sys

print(__doc__)
sys.exit(
    "fix_data.py is deprecated (full non-streaming download) — "
    "run: python setup_scripts/stream_data.py"
)
