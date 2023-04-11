# Copyright 2023 Google.
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

r"""Extract subsampled vocabulary embeddings from a t5x checkpoint and save it as a numpy file.

The numpy file can subsequently be loaded as initial prompt embeddings (for
example, using configs/prompts/from_file.gin).

Example usage:
python -m prompt_tuning.scripts.subsample_vocab \
  --checkpoint_dir=/path/to/t5x/checkpoint_step \
  --embeddings_path=target/encoder/prompt/prompt/prompt \
  --restore_dtype=float32 \
  --prompt_length=50 \
  --output_path=/path/to/save/subsampled_vocab.npy

"""

import os
import re
from typing import Mapping, Any, Sequence
from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
import numpy as np
from t5x import checkpoints
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_dir", None, "The path to the t5x checkpoint directory")
flags.DEFINE_string(
    "embeddings_path",
    "target/encoder/prompt/prompt/prompt",
    (
        "The path to the vocabulary embeddings in the checkpoint tree, using"
        " `/` for scoping. Leading `/` or `/target` is optional."
    ),
)
flags.DEFINE_enum(
    "restore_dtype",
    "float32",
    ["float32", "bfloat16"],
    "The data type to use when restoring the embeddings.")
flags.DEFINE_string(
    "output_path",
    None,
    "The path to where the numpy embeddings should be saved.")
flags.DEFINE_integer(
    "prompt_length",
    100,
    "Number of samples to keep (sampled with replacement).",
    lower_bound=1,
)
flags.DEFINE_integer("population_size", 5000,
                     "Limit the drawing to the first `population_size` vectors."
                     " If <= 0 then there is no limit."
                     )
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.mark_flag_as_required("checkpoint_dir")
flags.mark_flag_as_required("output_path")


def normalize_variable_path(path: str, sep: str = "/") -> str:
  """Make sure path starts with `target/`."""
  # TODO: enable saving all variables within a scope if the path
  # ends in the separator.
  path = path.strip(sep)
  path = re.sub(r"^target/", "", path)
  return f"target/{path}"


def extract_nested_key(
    nested_key: str, blob: Mapping[str, Any], sep: str = "/") -> Any:
  """Extract a key nested dicts using a scoping separator."""
  # TODO: Add nicer error handling that shows where in the nested
  # dicts your key lookup fails.
  for key in nested_key.split(sep):
    blob = blob[key]
  return blob


def save_variable(output_path: str, variable: np.ndarray):
  """Save variable at output path using numpy."""
  dir_name = os.path.dirname(output_path)
  if not gfile.exists(dir_name):
    gfile.makedirs(dir_name)

  with gfile.GFile(output_path, "wb") as wf:
    np.save(wf, variable)


def main(argv: Sequence[str]):
  """Extract a numpy value from a t5x checkpoint."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line-arguments.")

  restore_dtype = jnp.dtype(FLAGS.restore_dtype)

  checkpoint = checkpoints.load_t5x_checkpoint(
      FLAGS.checkpoint_dir,
      restore_dtype=restore_dtype,
      lazy_parameters=True)

  logging.info("Reading variables from %s as dtype=%s",
               FLAGS.checkpoint_dir,
               restore_dtype)

  variable_path = normalize_variable_path(FLAGS.embeddings_path)
  logging.info("Extracting variable found at %s", variable_path)

  variable = extract_nested_key(variable_path, checkpoint)
  variable = variable.get()
  logging.info("Read variable with shape %s", variable.shape)

  logging.info("Saving variable to %s", FLAGS.output_path)

  rng: np.random.Generator = np.random.default_rng(FLAGS.seed)
  if FLAGS.population_size > 0:
    indices = rng.choice(FLAGS.population_size, FLAGS.prompt_length)
    variable = variable[indices]
  else:
    variable = rng.choice(variable, FLAGS.prompt_length)

  save_variable(FLAGS.output_path, variable)


if __name__ == "__main__":
  app.run(main)
