#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
#
# ./scripts/train_nucleui.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR='./data/1-nuclei/models/nucleui-models2'
DATASET_DIR='./data/1-nuclei/images'

python step4_train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=nuclei \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=nuclei \
  --preprocessing_name=cifarnet \
  --max_number_of_steps=100000 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

TRAIN_DIR='./data/1-nuclei/models/nucleui-models2'
DATASET_DIR='./data/1-nuclei/images'
CHECKPOINT_DIR='./data/1-nuclei/models/nucleui-models2/model.ckpt-7745'
EVAL_DIR='./data/1-nuclei/models/nucleui-models2/eval/'

# Run evaluation.
python step5_eval_image_classifier.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --eval_dir=${EVAL_DIR} \
  --dataset_name=nuclei \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --preprocessing_name=cifarnet \
  --model_name=nuclei

# Segment test image.
python step6_segment_test_images.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --preprocessing_name=cifarnet \
  --model_name=nuclei  \
  --infile=./data/1-nuclei/test_w32_parent_1.txt

