"""Keep results and caches on Google Drive so a Colab disconnect costs nothing.

Colab runtimes drop -- free ones after ~90 min idle, and any of them at 12 h.
The scale-up sweeps run for hours, so anything written only to /content is at
risk.  Work happens on the fast local disk and is mirrored to Drive after each
stage; on restart `pull()` brings it back and the grid's own jsonl caching
skips everything already done.

Two directories matter:
  results/  -- ~15 MB, the actual output, mirrored after every stage
  .cache/   -- ~380 MB of corpus embeddings and PCA pools; expensive to rebuild
               (~5 min) but not worth re-uploading, so it lives on Drive too

Usage inside the notebook:

    from colab.sync import pull, push, drive_dir
    pull()                       # restore from Drive, if anything is there
    ... run a stage ...
    push()                       # mirror back
"""

from __future__ import annotations

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
MIRROR = ("results", ".cache", "figures")
DEFAULT_DRIVE = "/content/drive/MyDrive/pe_toy"


def drive_dir(path: str = DEFAULT_DRIVE) -> pathlib.Path | None:
    """The Drive-backed store, or None if Drive is not mounted."""
    p = pathlib.Path(path)
    if not pathlib.Path("/content/drive/MyDrive").exists():
        return None
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_tree(src: pathlib.Path, dst: pathlib.Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if f.is_dir() or "__pycache__" in f.parts:
            continue
        target = dst / f.relative_to(src)
        if target.exists() and target.stat().st_size == f.stat().st_size:
            continue  # already mirrored, same size -> skip the FUSE write
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)


def pull(path: str = DEFAULT_DRIVE, verbose=True):
    """Restore results/.cache/figures from Drive into the working tree."""
    d = drive_dir(path)
    if d is None:
        if verbose:
            print("Drive not mounted -- nothing to restore")
        return False
    for name in MIRROR:
        src = d / name
        if src.exists():
            _copy_tree(src, ROOT / name)
            n = sum(1 for _ in (ROOT / name).rglob("*") if _.is_file())
            if verbose:
                print(f"  pulled {name}: {n} files")
    return True


def push(path: str = DEFAULT_DRIVE, verbose=True):
    """Mirror results/.cache/figures to Drive."""
    d = drive_dir(path)
    if d is None:
        if verbose:
            print("Drive not mounted -- results stay on /content and will be "
                  "lost when the runtime recycles")
        return False
    for name in MIRROR:
        src = ROOT / name
        if src.exists():
            _copy_tree(src, d / name)
            if verbose:
                print(f"  pushed {name}")
    return True


def status(path: str = DEFAULT_DRIVE):
    d = drive_dir(path)
    print(f"working tree : {ROOT}")
    print(f"drive store  : {d if d else 'NOT MOUNTED'}")
    for name in MIRROR:
        local = ROOT / name
        n_local = sum(1 for _ in local.rglob("*") if _.is_file()) if local.exists() else 0
        n_drive = 0
        if d and (d / name).exists():
            n_drive = sum(1 for _ in (d / name).rglob("*") if _.is_file())
        print(f"  {name:10s} local {n_local:5d} files | drive {n_drive:5d} files")
    for f in sorted((ROOT / "results").glob("*.jsonl")) if (ROOT / "results").exists() else []:
        print(f"    {f.name:34s} {sum(1 for _ in f.open()):5d} runs")
