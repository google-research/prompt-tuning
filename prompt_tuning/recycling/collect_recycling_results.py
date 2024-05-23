# Copyright 2024 Google.
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

"""Collect results for recycling experiments.

Note:
  CSV column ordering depends on insertion ordering when iterating through
  dicts, guaranteed in python>=3.7 and practically supported in cpython>=3.6.

  See prompt_tuning.recycling.recycle for a summary of the config file
  format.
"""

import csv
import io
import itertools
import json
import os
import re
from typing import Optional, Mapping, Any, Set
from absl import app
from absl import flags
from absl import logging
import pandas as pd
from tensorflow.io import gfile


DEFAULT_CONFIG = "./prompt_tuning/recycling/configs/sst2.json"

_CONFIG_FILE = flags.DEFINE_string(
    "config",
    DEFAULT_CONFIG,
    "Configuration file with all the settings and information.")
_TRAINING = flags.DEFINE_boolean(
    "training",
    True,
    "Should we collect training results?")
_EVALUATION = flags.DEFINE_boolean(
    "evaluation",
    True,
    "Should we collect evaluation results?")


def read_jsonl(path: str):
  """Read a jsonl file (one json object per line)."""
  with gfile.GFile(path) as rf:
    return [json.loads(l) for l in rf]


def fetch_training_metrics(
    model_dir: str,
    task_name: str,
    metric_name: str = "accuracy",
    display_metric: str = "Acc",
    constant_fields: Optional[Mapping[str, Any]] = None,
    steps: Optional[Set[int]] = None
) -> str:
  """Get training metrics and format as a TSV.

  The training metrics are expected to live in a single file with multiple steps
  per file.

  Args:
    model_dir: Where the model we are getting results from was saved.
    task_name: What task are we extracting training metrics from.
    metric_name: The field name what is in the jsonl file.
    display_metric: The field name we use in the TSV file.
    constant_fields: Extra values that are added to each line of the CSV.
    steps: A se of step values to collect results for.

  Returns:
    The training data formatted as a TSV string.
  """
  if constant_fields is None:
    constant_fields = {}
  if steps is None:
    steps = set()
  metric_path = os.path.join(model_dir,
                             "inference_eval",
                             f"{task_name}-metrics.jsonl")
  data = read_jsonl(metric_path)
  with io.StringIO() as wf:
    writer = csv.DictWriter(
        wf,
        fieldnames=list(itertools.chain(
            constant_fields.keys(),
            ["Step", display_metric]
        )),
        dialect="unix",
        delimiter="\t",
        quoting=csv.QUOTE_MINIMAL)
    for d in data:
      # If we don't have a step filter, or if we do and the step passes the
      # filter, create a row for this step.
      if not steps or d["step"] in steps:
        writer.writerow({
            **constant_fields,
            "Step": d["step"],
            display_metric: d[metric_name]
        })
    wf.seek(0)
    return wf.read()


SEED_REGEX = r"Large|Default|Seed_\d+"
EVAL_DIR_RE = re.compile(fr"""
  ^recycled_from_(?P<source_model>{SEED_REGEX})_at_(?P<step>\d+)_to_(?P<target_model>{SEED_REGEX})
""", re.VERBOSE)


def fetch_eval_metrics(
    eval_dir: str,
    task_name: str,
    metric_name: str = "accuracy",
    display_metric: str = "Acc",
    constant_fields: Optional[Mapping[str, Any]] = None,
    steps: Optional[Set[int]] = None
) -> str:
  """Collect recycling evaluation results from a single setting.

  For evaluation metrics, this function expects a directory that contains
  multiple recycling evaluation runs, each living in a directory named
    `recycled_from_${source_model}_at_${step}_to_${target_model}`

  Information like source model, target model, and step are extracted from the
  directory name that holds the recycling experiment while settings like init,
  run, and recycling method are contained in the directory hierarchy.

  Args:
    eval_dir: The directory that holds multiple evaluation runs.
    task_name: What task are we extracting training metrics from.
    metric_name: The field name what is in the jsonl file.
    display_metric: The field name we use in the TSV file.
    constant_fields: Extra values that are added to each line of the CSV.
    steps: A se of step values to collect results for.

  Returns:
    The recycling eval results as a TSV.
  """
  if constant_fields is None:
    constant_fields = {}

  with io.StringIO() as wf:
    writer = csv.DictWriter(
        wf,
        fieldnames=list(itertools.chain(
            constant_fields.keys(),
            ["Source Model", "Target Model", "Step", display_metric])),
        dialect="unix",
        delimiter="\t",
        quoting=csv.QUOTE_MINIMAL)
    if not gfile.exists(eval_dir):
      logging.warning("Eval dir %s not found!", eval_dir)
      return ""
    eval_dirs = gfile.listdir(eval_dir)
    logging.info(
        "Extracting up to %d recycling metrics from %s",
        len(eval_dirs),
        eval_dir)
    for eval_d in eval_dirs:
      if m := EVAL_DIR_RE.match(eval_d):
        if not steps or int(m.group("step")) in steps:
          metric_file = os.path.join(
              eval_dir,
              eval_d,
              "inference_eval",
              f"{task_name}-metrics.jsonl")
          if not gfile.exists(metric_file):
            logging.warning("Metric file %s not written yet", metric_file)
            continue
          # There is only 1 value in the recycled metrics
          metrics = read_jsonl(metric_file)[0]
          writer.writerow({
              **constant_fields,
              "Source Model": m.group("source_model").replace("_", " "),
              "Target Model": m.group("target_model").replace("_", " "),
              "Step": m.group("step"),
              display_metric: metrics[metric_name]
          })
    wf.seek(0)
    return wf.read()


def main(argv):
  """Collect metrics from our recycing experiments into a single TSV."""
  del argv

  with gfile.GFile(_CONFIG_FILE.value) as rf:
    config = json.load(rf)

  steps = set(config.get("steps", set()))
  dataset = config["dataset"]
  task_name = config["task_name"]

  # Collect Training Metrics
  if _TRAINING.value:
    rows = []
    for seed in config["prompts"]:
      for init in config["prompts"][seed]:
        for run in config["prompts"][seed][init]:
          # Read and format each section as a TSV string as it can be piecemeal
          # before turning the full string into a pandas DataFrame.
          rows.append(fetch_training_metrics(
              config["prompts"][seed][init][run],
              task_name,
              constant_fields={
                  "Model": seed,
                  "Init": init,
                  "Run": run,
              },
              steps=steps))

    header = "Model\tInit\tRun\tStep\tAcc"
    raw_data = "\n".join(itertools.chain([header], rows))
    with io.StringIO(raw_data) as rf:
      df = pd.read_csv(rf, sep="\t")

    training_tsv = os.path.join(config["root_dir"], f"training-{dataset}.tsv")
    logging.info(
        "Writing %d rows of training metrics to %s", len(df), training_tsv)

    with gfile.GFile(training_tsv, "w") as wf:
      df.to_csv(wf, index=False, sep="\t")

  # Collect Recycling Metrics
  if _EVALUATION.value:
    some_seed = list(config["prompts"])[0]
    inits = set(config["prompts"][some_seed])
    runs = set(config["prompts"][some_seed][list(inits)[0]])
    methods = set(config["recycling_methods"])

    rows = []
    for method in methods:
      for init in inits:
        for run in runs:
          eval_dir = os.path.join(
              config["root_dir"], method, config["dataset"], init, run, "eval"
          ).replace(" ", "_")
          # Read and format each section as a TSV string as it can be piecemeal
          # before turning the full string into a pandas DataFrame.
          rows.append(fetch_eval_metrics(
              eval_dir,
              task_name,
              constant_fields={
                  "Recycler": method,
                  "Init": init,
                  "Run": run
              },
              steps=steps))
    header = "Recycler\tInit\tRun\tSource Model\tTarget Model\tStep\tAcc"
    raw_data = "\n".join(itertools.chain([header], rows))
    with io.StringIO(raw_data) as rf:
      df = pd.read_csv(rf, sep="\t")

    eval_tsv = os.path.join(config["root_dir"], f"recycling-{dataset}.tsv")
    logging.info(
        "Writing %d recycling experiment results to %s", len(df), eval_tsv)
    with gfile.GFile(eval_tsv, "w") as wf:
      df.to_csv(wf, index=False, sep="\t")


if __name__ == "__main__":
  app.run(main)
