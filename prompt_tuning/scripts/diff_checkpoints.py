# Copyright 2022 Google.
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

r"""Diff two t5x checkpoints.

Usage:
python3 -m prompt_tuning.scripts.diff_checkpoints \
  /path/to/checkpoint/a
  /path/to/checkpoint/b
  --frac 0.1
  --limit_to_params
  --output /tmp/a_vs_b.diff

Above will diff checkpoints, a and b, by comparing 10% of their parameters.

"""

import json
import math
import random
import sys
from typing import Sequence, Dict, Optional, Any
from absl import app
from absl import flags
from absl import logging
from flax import traverse_util
import numpy as np
from t5x import checkpoint_importer as ckpt_impt
from t5x import checkpoints
from tensorflow.io import gfile


FLAGS = flags.FLAGS
flags.DEFINE_float("frac", 1.0, "The fraction of parameters to check (0, 1].")
flags.DEFINE_string("output", None, "Where to save a summary of the diff.")
flags.DEFINE_integer("seed",
                     None,
                     "A seed for parameter selection in the random spot check.")
flags.DEFINE_boolean("compare_step",
                     False,
                     "Should we check that the steps match?")
flags.DEFINE_boolean("limit_to_params",
                     False,
                     "Should we only compare parameter values?")
flags.DEFINE_multi_string("check", None, "A parameter we need to check.")


def validate_frac(value: float) -> bool:
  """Make sure --frac is in (0, 1]."""
  return 0.0 < value <= 1.0


def write_output(errors, output_path: str):
  """Save our diff to a file."""
  with gfile.GFile(output_path, "w") as wf:
    json.dump(errors, wf, indent=2)


def check_difference(a: ckpt_impt.LazyArray,
                     b: ckpt_impt.LazyArray) -> Optional[Dict[str, Any]]:
  """Compare two arrays and return a dict describing their differences."""
  a_none = bool(a is None)
  b_none = bool(b is None)
  if a_none and b_none:
    return None
  # XOR, only one is None
  if a_none ^ b_none:
    return {
        "error_type": "missing",
        "a_is_none": a_none,
        "b_is_none": b_none
    }
  if a.shape != b.shape:
    return {
        "error_type": "shape",
        "a_shape": a.shape,
        "b_shape": b.shape,
    }
  a = a.get()
  b = b.get()
  diff = np.isclose(a, b)
  if np.all(diff):
    return None
  total = np.prod(a.shape)
  match = np.sum(diff.astype(np.uint8))
  mismatch = total - match
  return {
      "error_type": "value",
      "match": match.item(),
      "mismatch": mismatch.item(),
      "percentage": f"{mismatch / total * 100:.3f}%",
      "max_error": np.max(np.abs(a - b)).item()
  }


def main(argv: Sequence[str]):
  if len(argv) != 3:
    raise ValueError("usage: python3 -m prompt_tuning.scripts.diff_checkpoints "
                     " /path/to/ckpt_a /path/to/ckpt_b")
  random.seed(FLAGS.seed)
  _, ckpt_a_path, ckpt_b_path = argv
  logging.info("Loading checkpoint %s", ckpt_a_path)
  ckpt_a = checkpoints.load_t5x_checkpoint(ckpt_a_path, lazy_parameters=True)
  ckpt_a = {"/".join(k): v
            for k, v in traverse_util.flatten_dict(ckpt_a).items()}
  logging.info("Loading checkpoint %s", ckpt_b_path)
  ckpt_b = checkpoints.load_t5x_checkpoint(ckpt_b_path, lazy_parameters=True)
  ckpt_b = {"/".join(k): v
            for k, v in traverse_util.flatten_dict(ckpt_b).items()}

  if FLAGS.limit_to_params:
    logging.info("Only comparing actual parameters, skipping optimizer state.")
    ckpt_a = {k: v for k, v in ckpt_a.items() if k.startswith("target/")}
    ckpt_b = {k: v for k, v in ckpt_b.items() if k.startswith("target/")}

  if not FLAGS.compare_step:
    logging.info("Removing state/step for easier cross checkpoint compares.")
    ckpt_a.pop("state/step", None)
    ckpt_b.pop("state/step", None)

  mismatched = False
  results = {
      "ckpt_a": ckpt_a_path,
      "ckpt_b": ckpt_b_path,
  }

  a_params = set(ckpt_a)
  b_params = set(ckpt_b)
  if a_params != b_params:
    a_minus_b = sorted(a_params - b_params)
    b_minus_a = sorted(b_params - a_params)
    logging.warning("Parameter Trees do not match!")
    logging.warning("Keys in ckpt_a and not in ckpt_b: %s", a_minus_b)
    logging.warning("Keys in ckpt_b and not in ckpt_a: %s", b_minus_a)
    results["parameter_mismatch"] = {
        "present_in_a_not_in_b": a_minus_b,
        "present_in_b_not_in_a": b_minus_a,
    }

  shared_parameters = a_params & b_params
  if FLAGS.frac != 1.0:
    k = math.floor(len(shared_parameters) * FLAGS.frac)
    parameters_to_check = set(random.sample(list(shared_parameters), k=k))
    if FLAGS.check:
      for parameter in FLAGS.check:
        if parameter not in shared_parameters:
          raise ValueError(f"Requested parameter to check {parameter} was not "
                           "found in the checkpoints.")
        parameters_to_check.add(parameter)
    logging.info("Spot Checking %d parameters. (%d/%d ≅ %f%%)",
                 len(parameters_to_check),
                 len(parameters_to_check),
                 len(shared_parameters),
                 math.floor(
                     len(parameters_to_check) / len(shared_parameters) * 100))
    parameters_to_check = sorted(parameters_to_check)
  else:
    logging.info("Checking all parameters.")
    parameters_to_check = sorted(shared_parameters)

  results["check_frac"] = FLAGS.frac
  results["parameters"] = {}
  results["errors"] = []
  for parameter in parameters_to_check:
    diff = check_difference(ckpt_a[parameter], ckpt_b[parameter])
    if diff is None:
      results["parameters"][parameter] = "Matched!"
      continue
    if diff["error_type"] == "missing":
      logging.warning("Parameter %s is missing from a checkpoint, "
                      "in a=%s, in b=%s",
                      parameter,
                      not diff["a_is_none"],
                      not diff["b_is_none"])
      results["parameters"][parameter] = ("Missing Parameter: In "
                                          f"a={not diff['a_is_none']}, in "
                                          f"b={not diff['b_is_none']}")
      results["errors"].append(diff)
      mismatched = True
    if diff["error_type"] == "shape":
      logging.warning("Parameter %s had a shape mismatch, %s vs. %s",
                      parameter,
                      diff["a_shape"],
                      diff["b_shape"])
      results["parameters"][parameter] = (f"Shape Mismatch: {diff['a_shape']} "
                                          f"vs. {diff['b_shape']}")
      results["errors"].append(diff)
      mismatched = True
    if diff["error_type"] == "value":
      logging.warning("Parameter %s had a value mismatch. %s mismatch, %f max.",
                      parameter,
                      diff["percentage"],
                      diff["max_error"])
      results["parameters"][parameter] = ("Value Mismatch: "
                                          f"{diff['percentage']} mismatch, "
                                          f"{diff['max_error']:.4f} max.")
      results["errors"].append(diff)
      mismatched = True
  if FLAGS.output is not None:
    write_output(results, FLAGS.output)
  if mismatched:
    sys.exit(1)


if __name__ == "__main__":
  flags.register_validator(
      "frac",
      validate_frac,
      message="--frac must in (0, 1].",
      flag_values=FLAGS)
  app.run(main)
