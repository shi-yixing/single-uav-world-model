#!/bin/bash
# Usage: bash run_rl.sh <logdir> <map_scale> [extra train_rl.py args]
# Starts the Isaac env server, waits until it listens, then runs DreamerV3 training against it.
source /media/gsk/disk4t/yixing/env.sh
cd $Y/wm_data
LOGDIR=$1; SCALE=$2; shift 2
PORT=${PORT:-6100}
mkdir -p $LOGDIR
LABELS=""; [ "$(echo "$SCALE > 0" | bc)" = 1 ] && LABELS="--labels"
conda activate $Y/envs/isaac41
python rl_server.py --num_envs ${ENVS:-16} --port $PORT $LABELS ${SERVER_ARGS:-} > $LOGDIR/server.log 2>&1 &
SERVER=$!
until grep -q SERVER_READY $LOGDIR/server.log; do
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -20 $LOGDIR/server.log; exit 1; }
  sleep 5
done
conda activate $Y/envs/dreamer
export PYTHONPATH=$Y/wm_data:$PYTHONPATH XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.45}
python train_rl.py --logdir $LOGDIR --map_scale $SCALE --envs ${ENVS:-16} --port $PORT "$@" > $LOGDIR/train.log 2>&1
STATUS=$?
kill $SERVER 2>/dev/null
exit $STATUS
