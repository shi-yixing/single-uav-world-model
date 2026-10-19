#!/bin/bash
# M3/M4: online DreamerV3, baseline then map loss (lambda=30); 50M model, 1.5M env steps each, same seed.
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
for cfg in "rl_base 0 6130" "rl_map30 30 6131"; do
  set -- $cfg
  [ -f $Y/runs/$1/.done ] && continue
  ENVS=16 MEMFRAC=0.5 PORT=$3 bash run_rl.sh $Y/runs/$1 $2 --size size50m --steps 1.5e6 --train_ratio 64 --replay_size 1e6 --seed 0 || { echo "$1 failed"; exit 1; }
  touch $Y/runs/$1/.done
  echo "$1 done"
done
echo PIPELINE_RL_DONE
