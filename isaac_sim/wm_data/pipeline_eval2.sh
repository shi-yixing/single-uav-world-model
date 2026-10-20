#!/bin/bash
# Re-run evaluation with the lock-step fix (eval_rl.py) and the post-reset render flush (rl_server.py).
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
for cfg in "rl_base 0 6170" "rl_map30 30 6171"; do
  set -- $cfg
  LABELS=""; [ "$2" != "0" ] && LABELS="--labels"
  mv $Y/runs/$1/eval $Y/runs/$1/eval_v1 2>/dev/null
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
echo PIPELINE_EVAL2_DONE
