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
#
# Usage:
#
# ./scripts/train_nucleui.sh
set -e


TRAIN_DIR='./data/1-nuclei/models/nucleui-models-600K'
DATASET_DIR='./data/1-nuclei/images'

python step4_train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=nuclei \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=nuclei \
  --preprocessing_name=nuclei \
  --max_number_of_steps=600000 \
  --batch_size=128 \
  --save_interval_secs=600 \
  --save_summaries_secs=600 \
  --log_every_n_steps=100 \
  --optimizer=adagrad \
  --learning_rate=0.001 \
  --weight_decay=0.004




CHECKPOINT_DIR='./data/1-nuclei/models/nucleui-models-600K/model.ckpt-600000'
DATASET_DIR='./data/1-nuclei/images'
EVAL_DIR='./data/1-nuclei/models/nucleui-models-600K/eval/'

#measure performance 
python step5_eval_image_classifier.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --eval_dir=${EVAL_DIR} \
  --dataset_name=nuclei \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --preprocessing_name=nuclei \
  --model_name=nuclei 



#generate segmentation image
python step6_segment_test_images.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --preprocessing_name=nuclei \
  --model_name=nuclei   \
  --infile=./data/1-nuclei/test_w32_parent_1.txt
