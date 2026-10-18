#!/bin/bash
# Step 2 (offline): world models with the auxiliary map loss, then probe against the plain baseline.
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
conda activate $Y/envs/dreamer
for S in 1 10; do
  RUN=$Y/runs/wm50m_map$S
  python train_wm.py train --data $Y/data/wm_v1 --logdir $RUN --size size50m --steps 20000 --map_scale $S || exit 1
  python train_wm.py extract --data $Y/data/wm_v1 --logdir $RUN --size size50m --map_scale $S || exit 1
done
conda activate $Y/envs/isaac41
OUT=$Y/runs/step2_probe; mkdir -p $OUT; ln -sfn $Y/runs/wm50m/probe/label_cache $OUT/label_cache
python probe.py --data $Y/data/wm_v1 --feats base=$Y/runs/wm50m/features map1=$Y/runs/wm50m_map1/features \
  map10=$Y/runs/wm50m_map10/features --out $OUT || exit 1
echo PIPELINE3_DONE
