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
Preprocess the CF dataset to parquet format
"""

import os
import re
import argparse
import datasets
import numpy as np

from verl.utils.hdfs_io import copy, makedirs


def load_data_from_path(csv_path: str) -> datasets.Dataset:
    return datasets.load_dataset("csv", data_files=csv_path) if csv_path is not None else None

def make_map_fn(split, use_tool):
    def transform_examples(examples_list):
        """Converts list-of-strings inputs/outputs to single strings."""
        transformed = []
        for ex in examples_list:
            # Join input/output lists into single strings
            data = {
                "input": "".join(ex["input"]),
                "output": "".join(ex["output"])
            }
            if len(ex["explanation"]) > 0:
                data["explanation"] = ex["explanation"]
            transformed.append(data)
        return transformed

    def get_test_cases(examples_list):
        """Converts list-of-strings inputs/outputs to list of test cases."""

        inputs, outputs = [], []
        for ex in examples_list:
            inputs.append("".join(ex["input"]))
            outputs.append("".join(ex["output"]))
        return [{"input": inputs, "output": outputs}]
        
        
    def process_fn(example, idx):
        problem_statement = example.pop("statement")
        input_format = example.pop("input_format")
        output_format = example.pop("output_format")
        test_examples = example.pop("examples")
        problem_id = example.pop("problem_id")
        name = example.pop("name")
        data_source = example.pop("datasource")
        
        import ast
        formatted_test_examples = ast.literal_eval(test_examples)
        formatted_test_examples = transform_examples(formatted_test_examples)
        actual_tests = get_test_cases(formatted_test_examples)

        prompt = f"""Provide a Python solution to a competitive programming question.
        Problem statement:\n{problem_statement}\nInput specification:\n{input_format}\nOutput specification:\n{output_format}\nExamples:\n{formatted_test_examples[0]}\n
        
        Think step by step and write python code to solve this problem. 
        Present the code in ```python\nYour code\n```"""
        
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": prompt},
                
            ],
            "reward_model": {"style": "rule", "ground_truth": test_examples}, 
            "extra_info": {
                "index": idx,
                "name": name,
                "examples": actual_tests, 
                "problem_statement": problem_statement, 
                "problem_id": problem_id,
            }, 
        }

        # Only include tools_kwargs when --use_tool is True
        if use_tool:
            data["extra_info"]["need_tools_kwargs"] = True, 
            data["extra_info"]["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {"ground_truth": test_examples},
                }
            }
        return data
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--read_split", action="store_true")
    parser.add_argument("--language", default=None)
    parser.add_argument("--use_tool", action="store_true")

    args = parser.parse_args()

    if args.read_split:
        data_sources = [f"{args.csv_path}_train.csv", f"{args.csv_path}_test.csv", f"{args.csv_path}_val.csv"]
    else:
        data_sources = [f"{args.csv_path}.csv"]

    train_dataset = load_data_from_path(data_sources[0])
    test_dataset = load_data_from_path(data_sources[1] if len(data_sources) > 1 else None)
    dev_dataset = None
    # dev_dataset = load_data_from_path(data_sources[2] if len(data_sources) > 2 else None)

    # data.shuffle is already set to true in verl, but doing this for redundancy sake. 
    train_dataset = train_dataset.map(function=make_map_fn("train", args.use_tool), with_indices=True)
    train_dataset = train_dataset.shuffle(seed=42) 
    test_dataset = test_dataset.map(function=make_map_fn("test", args.use_tool), with_indices=True) if test_dataset is not None else None
    dev_dataset = dev_dataset.map(function=make_map_fn("dev", args.use_tool), with_indices=True) if dev_dataset is not None else None

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    # Helper function to save dataset with conditional "_tools" suffix
    def save_dataset(dataset, split_name):
        suffix = "_tools" if args.use_tool else ""
        filename = f"astra{split_name}_{args.language}{suffix}.parquet"
        output_path = os.path.join(local_dir, filename)
        
        if isinstance(dataset, datasets.DatasetDict):
            for _, sub_dataset in dataset.items():
                sub_dataset.to_parquet(output_path)
        else:
            dataset.to_parquet(output_path)

    save_dataset(train_dataset, "train")
    
    if test_dataset is not None:
        save_dataset(test_dataset, "test")
    
    if dev_dataset is not None:
        save_dataset(dev_dataset, "dev")
