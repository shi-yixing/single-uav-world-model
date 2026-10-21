#!/bin/bash
# Zero-shot evaluation of the round-3 policies on harder scenes (deadend_long, nested3, fork).
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
for cfg in "rl3_base 0 6240" "rl3_map30 30 6241"; do
  set -- $cfg
  LABELS=""; [ "$2" != "0" ] && LABELS="--labels"
  conda activate $Y/envs/isaac41
  python rl_server.py --num_envs 16 --port $3 $LABELS --extras --seed 7 > $Y/runs/$1/hard_server.log 2>&1 &
  SERVER=$!
  until grep -q SERVER_READY $Y/runs/$1/hard_server.log; do sleep 5; done
  conda activate $Y/envs/dreamer
  mv $Y/runs/$1/eval $Y/runs/$1/eval_round3
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python eval_rl.py --logdir $Y/runs/$1 --map_scale $2 --port $3 --episodes 100 \
    --suites deadend_long,nested3,fork --seed_base 400000 > $Y/runs/$1/hard_eval.log 2>&1 || { echo "$1 eval failed"; kill $SERVER; mv $Y/runs/$1/eval_round3 $Y/runs/$1/eval; exit 1; }
  mv $Y/runs/$1/eval $Y/runs/$1/eval_hard; mv $Y/runs/$1/eval_round3 $Y/runs/$1/eval
  kill $SERVER; sleep 10
  echo "== $1"; grep EVAL $Y/runs/$1/hard_eval.log
done
echo HARD_EVAL_DONE
