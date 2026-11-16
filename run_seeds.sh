#!/bin/bash
# 5 BiLSTMs (100 epochs, u_in running total), seeds 1-5, same holdout split; then average.
set -e
for s in 1 2 3 4 5; do
  python3 -I lstm.py --epochs 100 --u-in-cumsum --seed $s \
    --trial-name bilstm_h128_l3_e100_cumsum_seed$s \
    --save-holdout-preds holdout_preds/seed$s.npz 2>&1 | tee lstm_h128_l3_e100_cumsum_seed$s.log
done
python3 -I ensemble_holdout.py --name bilstm_h128_l3_e100_cumsum_seeds1-5_mean \
  holdout_preds/seed{1,2,3,4,5}.npz 2>&1 | tee ensemble_seeds1-5.log
