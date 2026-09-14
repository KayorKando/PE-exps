"""Unpack the uploaded bundle and install deps. Short and chatty, so it fits
inside `colab exec`'s output timeout; everything slow is a detached task."""
import os
import subprocess
import sys
import zipfile

BUNDLE, WORK = "/content/pe_toy_bundle.zip", "/content/toyexp"
if not os.path.exists(BUNDLE):
    raise SystemExit(f"{BUNDLE} missing -- `colab upload` it first")
os.makedirs(WORK, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall(WORK)
print(f"[unpack] -> {WORK}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers"], check=True)
import multiprocessing as mp
print(f"[unpack] deps ok | cores={mp.cpu_count()}")
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)
