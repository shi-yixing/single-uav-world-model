#!/bin/bash
# Stop collection at 1000 episodes (GPU now shared with another job), then train / extract / probe.
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
until [ $(ls $Y/data/wm_v1/ep_*.npz | wc -l) -ge 1000 ] || ! pgrep -f collect.py >/dev/null; do sleep 30; done
pkill -INT -f "collect.py --num_envs 16" ; sleep 20; pkill -9 -f "collect.py --num_envs 16"
echo "collection stopped at $(ls $Y/data/wm_v1/ep_*.npz | wc -l) episodes"
RUN=$Y/runs/wm50m
conda activate $Y/envs/dreamer
python train_wm.py train --data $Y/data/wm_v1 --logdir $RUN --size size50m --steps 20000 || exit 1
python train_wm.py extract --data $Y/data/wm_v1 --logdir $RUN --size size50m || exit 1
python train_wm.py extract --data $Y/data/wm_v1 --logdir $RUN --size size50m --random_init --feat_dir features_rand || exit 1
conda activate $Y/envs/isaac41
python probe.py --data $Y/data/wm_v1 --feats wm=$RUN/features rand=$RUN/features_rand --out $RUN/probe || exit 1
echo PIPELINE_DONE
