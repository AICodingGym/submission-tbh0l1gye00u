#!/bin/bash
# 5 BiLSTMs (100 epochs, u_in running total), seeds 1-5, same holdout split; then average.
#   ./run_seeds.sh          -> original setup (inputs: 5 raw + u_in running total)
#   ./run_seeds.sh lags     -> also --u-in-lags (u_in lag 1, lead 1, diff 1)
# Holdout runs never write submission.csv.
set -e
EXTRA=""
TAG="cumsum"
if [ "$1" = "lags" ]; then
  EXTRA="--u-in-lags"
  TAG="cumsum_lags"
fi
for s in 1 2 3 4 5; do
  python3 -I lstm.py --epochs 100 --u-in-cumsum $EXTRA --seed $s \
    --trial-name bilstm_h128_l3_e100_${TAG}_seed${s}_ensmember \
    --save-holdout-preds holdout_preds/${TAG}_seed$s.npz 2>&1 | tee lstm_h128_l3_e100_${TAG}_seed$s.log
done
python3 -I ensemble_holdout.py --name bilstm_h128_l3_e100_${TAG}_seeds1-5_mean \
  holdout_preds/${TAG}_seed{1,2,3,4,5}.npz 2>&1 | tee ensemble_${TAG}_seeds1-5.log
