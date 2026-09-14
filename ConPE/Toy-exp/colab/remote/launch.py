"""Spawn worker.py detached on the VM and return immediately.

Reads the task name from /content/task.txt (written by colab_run.sh just
before this runs). Refuses to start a second copy of a task already running.
"""
import os
import subprocess

WORK, LOGS = "/content/toyexp", "/content/logs"
os.makedirs(LOGS, exist_ok=True)
task = open("/content/task.txt").read().strip()
log = f"{LOGS}/{task}.log"
pidf = f"{LOGS}/{task}.pid"

if os.path.exists(pidf):
    try:
        os.kill(int(open(pidf).read().strip()), 0)
        raise SystemExit(f"[launch] {task} is already running "
                         f"(pid {open(pidf).read().strip()}) -- watch it instead")
    except (ProcessLookupError, ValueError):
        pass

with open(log, "ab") as fh:
    p = subprocess.Popen(
        ["python3", f"{WORK}/colab/remote/worker.py", task],
        cwd=WORK, stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True,       # survives the exec websocket closing
    )
open(pidf, "w").write(str(p.pid))
print(f"[launch] {task} started, pid {p.pid}, log {log}")
