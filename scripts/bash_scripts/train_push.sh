#!/usr/bin/env bash

python scripts/push_task/train_push.py \
--model-type PushCrossmodalKalmanFilter \
--experiment-name push_cmekf_measinit

python scripts/push_task/train_push.py \
--model-type PushKalmanFilter \
--experiment-name push_ekf_measinit

python scripts/push_task/train_push.py \
--model-type PushUnimodalKalmanFilter \
--experiment-name push_umekf_measinit

