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

Note:
  This script only recreates the model *parameters* at some given step, the
  optimizer state for the prompt variable at some step are not saved, thus the
  optimizer state in the final checkpoint are not rewound like the parameters
  are. This means this script can't be used to recreating an earlier
  checkpoint and restart training from a previous step.

"""

import os
from typing import List, Optional
from absl import logging
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from t5x import checkpoints
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from tensorflow.io import gfile


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


def recreate_checkpoint(model: models.BaseModel,
                        model_dir: str,
                        step: int,
                        concurrent_gb: int = 16,
                        save_dtype: jnp.dtype = jnp.float32,
                        output_dir: Optional[str] = None) -> None:
  """Add numpy checkpoint values at some step to a prompt tuning checkpoint."""
  # We are going to save the checkpoint with a new step number so if they don't
  # give a new place to put it, just use the same dir.
  if output_dir is None:
    output_dir = model_dir

  save_dtype = jnp.dtype(save_dtype)

  # Find the latest checkpoint
  latest_step = checkpoints.latest_step(model_dir)
  if latest_step is None:
    raise ValueError(f"Cannot find checkpoint directory in {model_dir}")
  logging.info("Loading checkpoint at step %d", latest_step)
  checkpoint_directory = checkpoints.get_checkpoint_dir(
      model_dir, latest_step)
  logging.info("Loading checkpoint from %s", checkpoint_directory)

  # Initialize the training state from the model which was loaded via gin.
  def initialize_train_state(rng):
    initial_variables = model.get_initial_variables(
        rng=rng,
        input_shapes={
            "encoder_input_tokens": (1, 1),
            "decoder_input_tokens": (1, 1)
        })
    return train_state_lib.FlaxOptimTrainState.create(model.optimizer_def,
                                                      initial_variables)

  train_state = jax.eval_shape(initialize_train_state, jax.random.PRNGKey(0))
  # We are doing this conversion on a single host with capped memory usage
  # so only 1 partitioned and keep params off devices.
  partitioner = partitioning.PjitPartitioner(1, params_on_devices=False)
  checkpointer = checkpoints.Checkpointer(
      train_state, partitioner, output_dir, save_dtype=save_dtype)
  train_state = checkpointer.restore(latest_step, lazy_parameters=True)

  # Flatten the parameters so we can overwrite them.
  flat_params = traverse_util.flatten_dict(train_state.params, sep="/")

  # Find the numpy files for the given step
  # TODO: Abstract this step to allow for more control of the
  # over-writing, i.e. given associated lists of variable overwrite paths and
  # paths to numpy files, overwrite several variables without being tied to
  # prompts produced by this training run.
  numpy_files = find_numpy_checkpoints(model_dir, step)
  if not numpy_files:
    raise ValueError(f"Cannot find any number checkpoints in {model_dir} "
                     f"with step {step}")
  # The numpy files for a checkpoint could include multiple parameters so
  # loop through them.
  for numpy_file in numpy_files:
    logging.info("Loading numpy variable from %s", numpy_file)
    # Load the numpy variable
    with gfile.GFile(numpy_file, "rb") as f:
      numpy_value = np.load(f).astype(checkpointer.restore_dtype)
    # Figure out where the variable goes in the pytree
    variable_path = file_name_to_variable_path(numpy_file)
    logging.info("Overwriting the variable at %s with the value loaded from %s",
                 variable_path, numpy_file)
    # Update the variable in the pytree
    flat_params[variable_path] = numpy_value

  train_state = train_state.replace_params(
      traverse_util.unflatten_dict(flat_params, sep="/"))
  # Set the step to the given step.
  logging.info("Setting step to %d", step)
  # TODO: This conversion of the train step from an int to a numpy
  # scalar should be upstreamed to the train_state as it currently allows one
  # to save the step as a python integer which causes an error when reloading.
  train_state = train_state.replace_step(np.array(step, dtype=save_dtype))

  logging.info("Saving result to %s", output_dir)
  # # Actually save the new checkpoint.
  checkpointer.save(train_state, concurrent_gb=concurrent_gb)


if __name__ == "__main__":
  # pylint:disable=g-import-not-at-top
  from absl import flags
  import gin
  from t5x import gin_utils
  # pylint:disable=g-import-not-at-to

  FLAGS = flags.FLAGS

  jax.config.parse_flags_with_absl()

  flags.DEFINE_multi_string(
      "gin_file",
      default=None,
      help="Path to gin configuration file. Multiple paths may be passed and "
      "will be imported in the given order, with later configurations  "
      "overriding earlier ones.")

  flags.DEFINE_multi_string(
      "gin_bindings", default=[], help="Individual gin bindings")

  flags.DEFINE_list(
      "gin_search_paths",
      default=["third_party/py/t5x/configs",
               "prompt_tuning/configs"],
      help="Comma-separated list of gin config path prefixes to be prepended "
      "to suffixes given via `--gin_file`. If a file appears in. Only the "
      "first prefix that produces a valid path for each suffix will be "
      "used.")

  def main(_):
    """True main function."""
    recreate_checkpoint_using_gin = gin.configurable(recreate_checkpoint)

    gin_utils.parse_gin_flags(FLAGS.gin_search_paths, FLAGS.gin_file,
                              FLAGS.gin_bindings)
    # Get gin-configured version of `recreate_checkpoint`.
    recreate_checkpoint_using_gin()

  gin_utils.run(main)
