#!/usr/bin/env python3
# save as: watch_progress.py
# run with: python watch_progress.py

import time
import os
from datetime import timedelta

STATE_FILE = "NistChemData/scripts/webbook-index/state.jsonl"
TOTAL = 144_795  # known total from NistChemData README

def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)

def fmt_eta(seconds):
    if seconds <= 0:
        return "unknown"
    return str(timedelta(seconds=int(seconds)))

start_count = count_lines(STATE_FILE)
start_time  = time.time()

print(f"Starting from {start_count:,} / {TOTAL:,} entries already done.")
print("Press Ctrl+C to stop watching.\n")

prev_count = start_count

try:
    while True:
        time.sleep(30)
        count     = count_lines(STATE_FILE)
        elapsed   = time.time() - start_time
        new_since = count - start_count
        pct       = 100.0 * count / TOTAL
        delta     = count - prev_count

        if new_since > 0:
            rate     = new_since / elapsed           # compounds/sec
            remaining = (TOTAL - count) / rate
            eta_str  = fmt_eta(remaining)
        else:
            eta_str = "waiting..."

        bar_len  = 40
        filled   = int(bar_len * count / TOTAL)
        bar      = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r[{bar}] {pct:5.2f}%  "
            f"{count:,}/{TOTAL:,}  "
            f"+{delta}/30s  "
            f"ETA: {eta_str}",
            end="", flush=True
        )
        prev_count = count

except KeyboardInterrupt:
    print("\nDone watching.")
