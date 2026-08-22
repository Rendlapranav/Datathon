"""
load_data.py — DEPRECATED, kept for reference only.

Superseded by setup_scripts/stream_data.py, which fixes two problems this
script had:
  1. Statistical validity: this script picked whichever ASIN happened to
     be the FIRST to accumulate TARGET_COUNT reviews while scanning the
     stream in order — an artifact of stream order, not the product with
     the most reviews overall.
  2. Memory: this script buffered full review dicts for EVERY ASIN it saw
     while scanning (asin_buckets), not just the eventual winner — on a
     multi-million-row scan that grows unbounded and is not laptop-safe.
     stream_data.py fixes this by counting with a Counter in pass 1
     (O(unique ASINs), no payloads) and only buffering rows for the small,
     already-decided set of flagship + competitor ASINs in pass 2.

Run setup_scripts/stream_data.py instead:
    python setup_scripts/stream_data.py

It also selects direct competitor products (same category, different
brand) for the benchmark comparison, which this script never did.
"""
import sys

print(__doc__)
sys.exit(
    "load_data.py is deprecated — run: python setup_scripts/stream_data.py"
)
