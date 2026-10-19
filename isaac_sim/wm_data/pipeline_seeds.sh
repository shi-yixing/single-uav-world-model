#!/bin/bash
# Step 2 follow-up: extra seeds for the baseline and lambda=10, plus lambda=30; then one joint probe run.
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
conda activate $Y/envs/dreamer
run() {  # name scale seed
  RUN=$Y/runs/$1
  [ -f $RUN/features/.done ] && return 0
  python train_wm.py train --data $Y/data/wm_v1 --logdir $RUN --size size50m --steps 20000 --map_scale $2 --seed $3 || exit 1
  python train_wm.py extract --data $Y/data/wm_v1 --logdir $RUN --size size50m --map_scale $2 --seed $3 || exit 1
  touch $RUN/features/.done
}
run wm50m_s1 0 1
run wm50m_map10_s1 10 1
run wm50m_s2 0 2
run wm50m_map10_s2 10 2
run wm50m_map30 30 0
conda activate $Y/envs/isaac41
OUT=$Y/runs/step2_seeds_probe; mkdir -p $OUT; ln -sfn $Y/runs/wm50m/probe/label_cache $OUT/label_cache
python probe.py --data $Y/data/wm_v1 --out $OUT --feats \
  base_s0=$Y/runs/wm50m/features base_s1=$Y/runs/wm50m_s1/features base_s2=$Y/runs/wm50m_s2/features \
  map10_s0=$Y/runs/wm50m_map10/features map10_s1=$Y/runs/wm50m_map10_s1/features map10_s2=$Y/runs/wm50m_map10_s2/features \
  map30_s0=$Y/runs/wm50m_map30/features || exit 1
echo PIPELINE4_DONE
