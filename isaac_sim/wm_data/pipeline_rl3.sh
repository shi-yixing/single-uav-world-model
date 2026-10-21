#!/bin/bash
# Runs unattended after the zero-shot check. Decision rule from memory_task_design.md:
#   both policies >= 90% on deadend and nested -> scenes too easy, stop (needs redesign)
#   otherwise -> round 3: retrain both with 30% trap/memory starts (utrap_in, ctrap, deadend, nested), then evaluate
# Results: runs/rl3_*/eval/*.json and logs/pipeline_rl3.log
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
until grep -q -E "MEMEVAL_DONE|failed" $Y/logs/memeval.log; do sleep 60; done
grep -q MEMEVAL_DONE $Y/logs/memeval.log || { echo "zero-shot eval failed"; exit 1; }
MIN=$(python3 - <<'PY'
import json
v = [json.load(open(f"/media/gsk/disk4t/yixing/runs/{r}/eval_memory/{s}.json"))["summary"]["goal"]
     for r in ("rl2_base", "rl2_map30") for s in ("deadend", "nested")]
print(min(v))
PY
)
echo "zero-shot minimum success: $MIN"
if python3 -c "import sys; sys.exit(0 if float('$MIN') >= 0.9 else 1)"; then
  echo "ZERO_SHOT_TOO_EASY: memory scenes need a harder design, round 3 not started"; exit 0
fi
for cfg in "rl3_base 0 6210" "rl3_map30 30 6211"; do
  set -- $cfg
  [ -f $Y/runs/$1/.done ] && continue
  SERVER_ARGS="--trap_start_frac 0.3 --trap_kinds utrap_in,ctrap,deadend,nested" ENVS=16 MEMFRAC=0.5 PORT=$3 \
    bash run_rl.sh $Y/runs/$1 $2 --size size50m --steps 1.5e6 --train_ratio 64 --replay_size 1e6 --seed 0 || { echo "$1 failed"; exit 1; }
  touch $Y/runs/$1/.done; echo "$1 done"
done
for cfg in "rl3_base 0 6220" "rl3_map30 30 6221"; do
  set -- $cfg
  LABELS=""; [ "$2" != "0" ] && LABELS="--labels"
  conda activate $Y/envs/isaac41
  python rl_server.py --num_envs 16 --port $3 $LABELS --extras --seed 7 > $Y/runs/$1/eval_server.log 2>&1 &
  SERVER=$!
  until grep -q SERVER_READY $Y/runs/$1/eval_server.log; do sleep 5; done
  conda activate $Y/envs/dreamer
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python eval_rl.py --logdir $Y/runs/$1 --map_scale $2 --port $3 --episodes 100 \
    --suites random,utrap_in,ctrap,deadend,nested --seed_base 300000 > $Y/runs/$1/eval.log 2>&1 || { echo "$1 eval failed"; kill $SERVER; exit 1; }
  kill $SERVER; sleep 10
  echo "== $1"; grep EVAL $Y/runs/$1/eval.log
done
echo PIPELINE_RL3_DONE
