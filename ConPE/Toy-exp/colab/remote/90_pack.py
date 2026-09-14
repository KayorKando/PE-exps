"""Zip results + figures to /content/results.zip for `colab download`.

Only those two directories -- zipping the whole tree would sweep in the 381 MB
.cache/ of corpus embeddings, which is rebuildable and not worth downloading.
"""
import os
import sys
import zipfile
sys.path.insert(0, "/content/toyexp/colab/remote")
WORK = "/content/toyexp"

import sys; sys.path.insert(0, WORK)
OUT = "/content/results.zip"
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("results", "figures"):
        base = os.path.join(WORK, sub)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, WORK))
                n += 1
print(f"[pack] {OUT}  {os.path.getsize(OUT) / 1e6:.1f} MB, {n} files")
rd = os.path.join(WORK, "results")
for f in sorted(os.listdir(rd)) if os.path.isdir(rd) else []:
    if f.endswith(".jsonl"):
        print(f"  results/{f:34s} {sum(1 for _ in open(os.path.join(rd, f))):5d} runs")
