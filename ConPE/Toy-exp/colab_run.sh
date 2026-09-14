#!/usr/bin/env bash
# Drive the experiment on a Colab VM from this terminal. The laptop stays idle.
#
#   ./colab_run.sh setup            provision VM, upload code, install deps
#   ./colab_run.sh start setup      build the config B caches   (~15 min)
#   ./colab_run.sh start grid       stages 0-5 + tables/figures (~1-2 h on 2 cores)
#   ./colab_run.sh start scaleup    configA_768 + nsyn_health   (~3 h)
#   ./colab_run.sh start q0scale    Q0 at N=1e6                 (~30 min)
#   ./colab_run.sh watch <task>     tail that task's log
#   ./colab_run.sh fetch            download results.zip here
#   ./colab_run.sh stop             release the VM  (ALWAYS -- idle VMs bill)
#
# Long tasks run detached ON the VM, so `colab exec`'s output timeout and any
# laptop disconnect are both harmless. `watch` is a short poll, safe to repeat.
#
# One-time auth (needs a browser, so run it yourself):
#   gcloud auth application-default login \
#     --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
# https://www.googleapis.com/auth/userinfo.email,\
# https://www.googleapis.com/auth/colaboratory
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
S="${COLAB_SESSION:-petoy}"
C=(colab --auth=adc)

settask() { echo "open('/content/task.txt','w').write('$1')" | "${C[@]}" exec -s "$S" >/dev/null; }

case "${1:-}" in
  setup)
    ./make_bundle.sh
    "${C[@]}" new -s "$S" 2>/dev/null || echo "[i] reusing existing session"
    "${C[@]}" upload -s "$S" pe_toy_bundle.zip /content/pe_toy_bundle.zip
    "${C[@]}" exec -s "$S" -f colab/remote/unpack.py ;;
  push)
    ./make_bundle.sh
    "${C[@]}" upload -s "$S" pe_toy_bundle.zip /content/pe_toy_bundle.zip
    "${C[@]}" exec -s "$S" -f colab/remote/unpack.py ;;
  start)
    settask "${2:?usage: start <setup|grid|scaleup|q0scale>}"
    "${C[@]}" exec -s "$S" -f colab/remote/launch.py ;;
  watch)
    settask "${2:?usage: watch <task>}"
    "${C[@]}" exec -s "$S" -f colab/remote/tail.py ;;
  fetch)
    "${C[@]}" exec -s "$S" -f colab/remote/90_pack.py
    "${C[@]}" download -s "$S" /content/results.zip ./results_from_colab.zip
    echo "[i] unpack with: unzip -o results_from_colab.zip" ;;
  stop)    "${C[@]}" stop -s "$S" ;;
  status)  "${C[@]}" status -s "$S"; "${C[@]}" sessions ;;
  *)       sed -n '2,17p' "$0"; exit 1 ;;
esac
