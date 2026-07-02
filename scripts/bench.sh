#!/usr/bin/env bash
# Append a benchmark run to benchmarks/history.tsv, correlated to the git commit
# (the RESULT line already carries git=<short-hash>), so benchmark movements can
# be traced back to the changes that caused them.
#
#   scripts/bench.sh [chappie args...]             # std build
#   FLAVOR=burn scripts/bench.sh [chappie args...] # burn build (adds Neocortex)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks

feat=()
[ "${FLAVOR:-std}" = "burn" ] && feat=(--features burn)

result=$(cargo run -q -p chappie-cli --bin chappie "${feat[@]}" -- "$@" 2>/dev/null | grep '^RESULT' || true)
[ -z "$result" ] && { echo "bench: no RESULT line produced" >&2; exit 1; }

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\t%s\n' "$ts" "$result" | tee -a benchmarks/history.tsv
