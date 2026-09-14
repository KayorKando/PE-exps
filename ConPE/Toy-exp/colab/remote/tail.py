"""Print the tail of a task log and whether it is still running."""
import os

LOGS = "/content/logs"
task = open("/content/task.txt").read().strip()
log, pidf = f"{LOGS}/{task}.log", f"{LOGS}/{task}.pid"

alive = False
if os.path.exists(pidf):
    try:
        os.kill(int(open(pidf).read().strip()), 0)
        alive = True
    except Exception:
        alive = False
print(f"[{task}] {'RUNNING' if alive else 'not running'}")
if os.path.exists(log):
    lines = open(log, errors="replace").read().splitlines()
    print(f"[{task}] {len(lines)} log lines, last 25:")
    for l in lines[-25:]:
        print("  " + l)
else:
    print(f"[{task}] no log yet")
