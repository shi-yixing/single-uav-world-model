#!/bin/bash
# M5: wait for both online runs, then evaluate each on the fixed test suites.
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
until grep -q -E "PIPELINE_RL_DONE|failed" $Y/logs/pipeline_rl.log; do sleep 120; done
grep -q PIPELINE_RL_DONE $Y/logs/pipeline_rl.log || { echo "training failed"; exit 1; }
for cfg in "rl_base 0 6140" "rl_map30 30 6141"; do
  set -- $cfg
  LABELS=""; [ "$2" != "0" ] && LABELS="--labels"
  conda activate $Y/envs/isaac41
  python rl_server.py --num_envs 16 --port $3 $LABELS --extras --seed 7 > $Y/runs/$1/eval_server.log 2>&1 &
  SERVER=$!
  until grep -q SERVER_READY $Y/runs/$1/eval_server.log; do sleep 5; done
  conda activate $Y/envs/dreamer
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python eval_rl.py --logdir $Y/runs/$1 --map_scale $2 --port $3 --episodes 100 \
    > $Y/runs/$1/eval.log 2>&1 || { echo "$1 eval failed"; kill $SERVER; exit 1; }
  kill $SERVER; sleep 10
  grep EVAL $Y/runs/$1/eval.log
done
echo PIPELINE_EVAL_DONE
