#!/bin/bash
# Zero-shot check: evaluate the round-2 policies on the long-memory scenes (no retraining).
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
for cfg in "rl2_base 0 6200" "rl2_map30 30 6201"; do
  set -- $cfg
  LABELS=""; [ "$2" != "0" ] && LABELS="--labels"
  conda activate $Y/envs/isaac41
  python rl_server.py --num_envs 16 --port $3 $LABELS --extras --seed 7 > $Y/runs/$1/memeval_server.log 2>&1 &
  SERVER=$!
  until grep -q SERVER_READY $Y/runs/$1/memeval_server.log; do sleep 5; done
  conda activate $Y/envs/dreamer
  mv $Y/runs/$1/eval $Y/runs/$1/eval_round2
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python eval_rl.py --logdir $Y/runs/$1 --map_scale $2 --port $3 --episodes 100 \
    --suites deadend,nested --seed_base 300000 > $Y/runs/$1/memeval.log 2>&1 || { echo "$1 eval failed"; kill $SERVER; exit 1; }
  mv $Y/runs/$1/eval $Y/runs/$1/eval_memory; mv $Y/runs/$1/eval_round2 $Y/runs/$1/eval
  kill $SERVER; sleep 10
  echo "== $1"; grep EVAL $Y/runs/$1/memeval.log
done
echo MEMEVAL_DONE
