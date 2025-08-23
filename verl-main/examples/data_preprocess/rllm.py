# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DeepScaleR-Preview-Dataset and AIME24 datasets to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/rllm")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load DeepScaleR-Preview-Dataset for training
    train_data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    train_dataset = datasets.load_dataset(train_data_source, split="train")

    # Load AIME24 for testing
    test_data_source = "math-ai/aime24"
    test_dataset = datasets.load_dataset(test_data_source, split="test")

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # Process training dataset
    def make_train_map_fn():
        def process_fn(example, idx):
            # Extract question and answer from DeepScaleR dataset
            # Adjust these field names based on the actual dataset structure
            question_raw = example.get("question", example.get("prompt", ""))
            answer_raw = example.get("answer", example.get("response", ""))
            
            question = question_raw + " " + instruction_following

            data = {
                "data_source": train_data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    # Process test dataset
    def make_test_map_fn():
        def process_fn(example, idx):
            # Extract question and answer from AIME24 dataset
            # Adjust these field names based on the actual dataset structure
            question_raw = example.get("question", example.get("prompt", ""))
            answer_raw = example.get("answer", example.get("response", ""))
            
            question = question_raw + " " + instruction_following

            data = {
                "data_source": test_data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_train_map_fn(), with_indices=True)
    test_dataset = test_dataset.map(function=make_test_map_fn(), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "deepscaler.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "aime24.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
