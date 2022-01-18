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

r"""Recreate a checkpoint from earlier in prompt tuning training.

When training using prompt tuning we generally set the number of checkpoints to
1. This gives us a checkpoint that has the right shape (the prompt variable, the
lack of optimizer state for the main model, etc) that can be used to restart
after a preemption. But because prompt tuning only updates ~0.003% of the
parameters for a T5 XXL model, it would be wasteful to save many checkpoints as
so much of the model is frozen. Instead, we emit a numpy checkpoint of just the
prompt.

The problem with the above is you always end up with the /last/ checkpoint
instead of the /best/ checkpoint. This binary is a tool to do checkpoint
surgery to splice the prompts numpy value for some given checkpoint into the
final checkpoint. A common usecase being the recreation the best performing
checkpoint to be used in things like the t5x `eval.py` script.

"""

import os
from typing import List, Sequence
from unittest import mock
from absl import app
from absl import flags
from absl import logging
import flax
from flax import optim
from flax import traverse_util
import jax.numpy as jnp
import numpy as np
from t5x import checkpoints
from t5x import partitioning
from t5x import state_utils
from t5x import train_state as train_state_lib
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir",
    None,
    "The model directory where the checkpoints were written.")
flags.DEFINE_integer(
    "step", None, "The step number of the prompt to load.")
flags.DEFINE_string("output_dir", None, "The output location.")
flags.DEFINE_integer(
    "concurrent_gb",
    16,
    "Maximum amount of memory to use at once, can use up to 4x in some cases.")
flags.DEFINE_enum(
    "save_dtype",
    "float32",
    ["float32", "bfloat16"],
    "The data type to use while saving.")
flags.DEFINE_enum(
    "restore_dtype",
    "float32",
    ["float32", "bfloat16"],
    "The datatype used while restoring.")
flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("step")


def find_numpy_checkpoints(model_dir: str, step: int) -> List[str]:
  """Find all numpy files for saved at `step` in `model_dir`."""
  checkpoint_dir = os.path.join(
      model_dir, "numpy_checkpoints", f"checkpoint_{step}")
  return [os.path.join(checkpoint_dir, f)
          for f in gfile.listdir(checkpoint_dir)]


def remove_suffix(s: str, suffix: str) -> str:
  """Remove `suffix` from the end of `s` if it exists."""
  if suffix and s.endswith(suffix):
    return s[:-len(suffix)]
  return s[:]


def file_name_to_variable_path(file_name: str) -> str:
  """Convert the "." separated checkpoint naming to "/" nesting."""
  variable_name = remove_suffix(os.path.basename(file_name), ".npy")
  return variable_name.replace(".", "/")


def build_optimizer(state_dict) -> optim.Optimizer:
  """Create an (Adafactor) optimizer that matches `state_dict`."""
  flat_state = state_utils.flatten_state_dict(
      state_dict["state"]["param_states"])
  none_keys = {f"/{k}" for k, v in flat_state.items() if v is None}

  def filter_fn(path, _):
    return path not in none_keys

  optimizer_def = optim.MultiOptimizer(
      (optim.ModelParamTraversal(filter_fn), optim.Adafactor()))
  with mock.patch.object(flax.optim.base, "isinstance") as is_instance_mock:
    is_instance_mock.side_effect = (
        lambda x, y: True if y == jnp.ndarray else isinstance(x, y))
    optimizer = optimizer_def.create(state_dict["target"])
  return optimizer.restore_state(state_dict)


def main(argv: Sequence[str]) -> None:
  """Add numpy checkpoint values at some step to a prompt tuning checkpoint."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line-arguments.")

  output_dir = FLAGS.output_dir
  if output_dir is None:
    output_dir = FLAGS.model_dir

  # Find the latest checkpoint
  latest_step = checkpoints.latest_step(FLAGS.model_dir)
  if latest_step is None:
    raise ValueError(f"Cannot find checkpoint directory in {FLAGS.model_dir}")
  logging.info("Loading checkpoint at step %d", latest_step)
  checkpoint_directory = checkpoints.get_checkpoint_dir(
      FLAGS.model_dir, latest_step)
  logging.info("Loading checkpoint from %s", checkpoint_directory)
  # Load the latest checkpoint
  checkpoint = checkpoints.load_t5x_checkpoint(
      checkpoint_directory,
      restore_dtype=FLAGS.restore_dtype,
      lazy_parameters=True)

  flat_checkpoint = state_utils.flatten_state_dict(
      checkpoint, keep_empty_nodes=True)

  # Find the numpy files for the given step
  # TODO: Abstract this step to allow for more control of the
  # over-writing, i.e. given associated lists of variable overwrite paths and
  # paths to numpy files, overwrite several variables without being tied to
  # prompts produced by this training run.
  numpy_files = find_numpy_checkpoints(FLAGS.model_dir, FLAGS.step)
  if not numpy_files:
    raise ValueError(f"Cannot find any number checkpoints in {FLAGS.model_dir} "
                     f"with step {FLAGS.step}")
  for numpy_file in numpy_files:
    logging.info("Loading numpy variable from %s", numpy_file)
    # Load the numpy variable
    with gfile.GFile(numpy_file, "rb") as f:
      numpy_value = np.load(f).astype(FLAGS.restore_dtype)
    # Figure out where the variable goes in the pytree
    variable_path = file_name_to_variable_path(numpy_file)
    logging.info("Overwriting the variable at %s with the value loaded from %s",
                 variable_path, numpy_file)
    # Update the variable in the pytree
    flat_checkpoint[f"target/{variable_path}"] = numpy_value
  # Set the step to the given step.
  logging.info("Setting step to %d", FLAGS.step)
  flat_checkpoint["state/step"] = np.asarray(
      FLAGS.step, dtype=flat_checkpoint["state/step"].dtype)

  # Save the checkpoint with the given step prompt included.
  checkpoint = traverse_util.unflatten_dict(
      {tuple(k.split("/")): v for k, v in flat_checkpoint.items()})
  partitioner = partitioning.PjitPartitioner(
      num_partitions=1, params_on_devices=False)
  # TODO: Add option to configure what optimizer to use.
  optimizer = build_optimizer(checkpoint)
  # Turn the optimizer into the train_state object
  train_state = train_state_lib.TrainState.from_flax_optimizer(optimizer)
  train_state = train_state.restore_state(checkpoint)
  checkpointer = checkpoints.Checkpointer(
      train_state,
      partitioner,
      output_dir,
      save_dtype=FLAGS.save_dtype,
      restore_dtype=FLAGS.restore_dtype)
  logging.info("Saving result to %s", output_dir)
  # Actually save the new checkpoint.
  checkpointer.save(train_state, concurrent_gb=FLAGS.concurrent_gb)


if __name__ == "__main__":
  app.run(main)
