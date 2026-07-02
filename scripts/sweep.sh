#!/usr/bin/env bash
# A longer benchmark sweep: run the sim across many seeds and a few ablation arms
# (turn each major system off in turn) to measure — with error bars — what each
# actually contributes, then one epic-length life. Robust to individual failures;
# writes incrementally so partial results survive an interruption.
#
#   scripts/sweep.sh                 # defaults below
#   SEEDS=100 TICKS=40000 scripts/sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-60}
TICKS=${TICKS:-30000}
EPIC=${EPIC:-20000000}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
OUT="benchmarks/sweep-${stamp}.tsv"
SUM="benchmarks/sweep-${stamp}.summary.md"

echo "building release binary..."
cargo build -q --release --bin chappie -p chappie-cli || { echo "build failed"; exit 1; }
BIN=./target/release/chappie

# ablation arms: name -> extra --set flags (empty = baseline / all systems on)
arm_names=(baseline no_growth no_gatekeeper no_thinking)
arm_flags=(""
           "--set growth.enabled=false"
           "--set gatekeeper.match_threshold=2.0"
           "--set thinking.max_escalations=0")

cols=(bench_final bench_auc bench_hard bench_recall reward thinks agents recruited reflexes days)
{ printf 'arm\tseed'; for c in "${cols[@]}"; do printf '\t%s' "$c"; done; printf '\n'; } > "$OUT"

echo "sweep: ${#arm_names[@]} arms x $SEEDS seeds x $TICKS ticks -> $OUT"
for i in "${!arm_names[@]}"; do
  arm="${arm_names[$i]}"; flags="${arm_flags[$i]}"
  for s in $(seq 1 "$SEEDS"); do
    line=$("$BIN" --seed "$s" --ticks "$TICKS" $flags 2>/dev/null | grep '^RESULT' || true)
    [ -z "$line" ] && { echo "  WARN $arm seed $s: no RESULT"; continue; }
    row="$arm	$s"
    for c in "${cols[@]}"; do
      v=$(grep -oP "(?<![a-z_])${c}=\K[-0-9.]+" <<<"$line" | head -1)
      row="$row	${v:-NA}"
    done
    printf '%s\n' "$row" >> "$OUT"
  done
  echo "  done arm: $arm"
done

echo "epic life: 1 seed x $EPIC ticks..."
epic=$("$BIN" --seed 1 --ticks "$EPIC" 2>/dev/null | grep '^RESULT' || echo "RESULT (epic failed)")

# ---- summary: mean ± std per arm per column ----
awk -F'\t' -v gh="$git_hash" '
NR==1 { for (i=3;i<=NF;i++) col[i]=$i; ncol=NF; next }
{
  a=$1; arms[a]=1
  for (i=3;i<=NF;i++) if ($i!="NA" && $i!="") { n[a,i]++; sum[a,i]+=$i; sq[a,i]+=$i*$i }
}
END {
  printf "# Sweep summary (git %s)\n\n", gh
  printf "mean ± std over seeds, per ablation arm.\n\n"
  printf "| arm |"; for (i=3;i<=ncol;i++) printf " %s |", col[i]; printf "\n"
  printf "|---|"; for (i=3;i<=ncol;i++) printf "---|"; printf "\n"
  na=split("baseline no_growth no_gatekeeper no_thinking", order, " ")
  for (k=1;k<=na;k++) { a=order[k]; if (!(a in arms)) continue
    printf "| %s |", a
    for (i=3;i<=ncol;i++) {
      c=n[a,i]; if (c>0) { m=sum[a,i]/c; var=sq[a,i]/c-m*m; if (var<0) var=0; sd=sqrt(var)
        printf " %.3f±%.3f |", m, sd } else printf " – |"
    }
    printf "\n"
  }
}' "$OUT" > "$SUM"

{
  echo ""
  echo "## Epic life (seed 1, $EPIC ticks)"
  echo '```'
  echo "$epic"
  echo '```'
} >> "$SUM"

echo "DONE. results: $OUT   summary: $SUM"
cat "$SUM"
