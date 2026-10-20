#!/bin/bash
# Round 2: 30% of training episodes start inside a U or C trap; evaluate on random / utrap_in / ctrap / utrap.
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
for cfg in "rl2_base 0 6180" "rl2_map30 30 6181"; do
  set -- $cfg
  [ -f $Y/runs/$1/.done ] && continue
  SERVER_ARGS="--trap_start_frac 0.3" ENVS=16 MEMFRAC=0.5 PORT=$3 bash run_rl.sh $Y/runs/$1 $2 --size size50m \
    --steps 1.5e6 --train_ratio 64 --replay_size 1e6 --seed 0 || { echo "$1 failed"; exit 1; }
  touch $Y/runs/$1/.done; echo "$1 done"
done
for cfg in "rl2_base 0 6190" "rl2_map30 30 6191"; do
  set -- $cfg
  LABELS=""; [ "$2" != "0" ] && LABELS="--labels"
  conda activate $Y/envs/isaac41
  python rl_server.py --num_envs 16 --port $3 $LABELS --extras --seed 7 > $Y/runs/$1/eval_server.log 2>&1 &
  SERVER=$!
  until grep -q SERVER_READY $Y/runs/$1/eval_server.log; do sleep 5; done
  conda activate $Y/envs/dreamer
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python eval_rl.py --logdir $Y/runs/$1 --map_scale $2 --port $3 --episodes 100 \
    --suites random,utrap_in,ctrap,utrap > $Y/runs/$1/eval.log 2>&1 || { echo "$1 eval failed"; kill $SERVER; exit 1; }
  kill $SERVER; sleep 10
  grep EVAL $Y/runs/$1/eval.log
done
echo PIPELINE_RL2_DONE
