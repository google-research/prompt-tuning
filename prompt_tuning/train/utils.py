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

"""Utilities for Prompt Tuning.

This module includes various utilities, mostly for working around the
limitations of gin.
"""

import os
import re
import time
from typing import Sequence, Callable, Any, Tuple, Optional
from absl import logging
import flax
from flax import optim
from flax import traverse_util
import jax
import numpy as np
from t5x import checkpoints
from t5x import multihost_utils
from t5x import partitioning
from t5x import train_state as train_state_lib
from tensorflow.io import gfile

PartitionRule = Tuple[str, Optional[partitioning.PartitionSpec]]


def match_any(regexes: Sequence[str]) -> Callable[[str, Any], bool]:
  """A traversal that checks if the parameter name matches any regex.

  This is returns a closure over the actual traversal function that takes the
  parameter name and value. The return value of this should be in input to the
  Traversal used in the MultiOptimizer.

  Args:
    regexes: A list of regular expressions that denote which parameter should be
      updated by this optimizer.

  Returns:
    A function that takes the name and value of a parameter and return True if
    that parameter should be updated by the optimizer.
  """
  regexes = tuple(re.compile(regex) for regex in regexes)

  def _match_any(path, _):
    """True if path matches any regex in regexs, false otherwise."""
    return any(regex.fullmatch(path) for regex in regexes)

  return _match_any


def np_save(path: str, arr) -> None:
  """Write a numpy file to disk, supporting remote file systems."""
  with gfile.GFile(path, 'wb') as wf:
    np.save(wf, arr)


class MultiOptimizer(optim.MultiOptimizer):
  """A MultiOptimizer subclass to support gin.

  Note:
    This optimizer does not work with the t5x version of adafactor and should
    only be used with a normal flax optimizer.

  Gin doesn't support binding to variadic parameters, like the
  `*traversals_and_optimizers` parameter for the MultiOptimizer. This subclass
  facilitates binding to this parameter by accepting a sequence as the value for
  a single parameter, which is when unpacked in the super call.
  """

  def __init__(
      self, traversals_and_optimizers: Sequence[Tuple[traverse_util.Traversal,
                                                      optim.OptimizerDef]]):
    super().__init__(*traversals_and_optimizers)


class Checkpointer(checkpoints.Checkpointer):
  """A checkpointer that saves some variables as numpy arrays first."""

  def __init__(self, *args, save_paths: Sequence[str], **kwargs):
    self.save_matcher = match_any(save_paths)
    super().__init__(*args, **kwargs)

  def save_numpy(
      self, targets: flax.core.frozen_dict.FrozenDict, step: int) -> None:
    # Save numpy values to
    # `${self.checkpoint_dir}/numpy_checkpoints/checkpoint_${step}`
    # Note, numpy values need to be saved apart from the default checkpoint so
    # that they are immune from the T5X checkpoint retention settings.
    numpy_dir = os.path.join(
        self.checkpoints_dir, 'numpy_checkpoints', f'checkpoint_{step}')
    # Write to a tmp dir to create an atomic checkpointing operation.
    timestamp = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
    tmp_dir = f'{numpy_dir}.tmp-{timestamp}'
    logging.info('Saving Numpy checkpoints for step %d to %s', step, tmp_dir)
    if jax.process_index() == 0:
      gfile.makedirs(tmp_dir)
    multihost_utils.sync_devices(f'checkpointer:save_numpy:make_dir:{tmp_dir}')

    if jax.process_index() == 0:
      # Save any variable whose name matches.
      flat_targets = {'/'.join(k): v
                      for k, v in traverse_util.flatten_dict(
                          flax.core.unfreeze(targets)).items()}

      for flat_name, value in flat_targets.items():
        if self.save_matcher(flat_name, None):
          dotted_name = flat_name.replace('/', '.')
          output_path = os.path.join(tmp_dir, dotted_name)
          np_save(output_path, np.array(value).astype(self._save_dtype))
    multihost_utils.sync_devices('checkpointer:save_numpy:writes_complete:'
                                 f'{tmp_dir}')
    if jax.process_index() != 0:
      return
    if gfile.exists(numpy_dir):
      backup_dir = f'{numpy_dir}.backup-{time.time()}'
      logging.info('%s already exists. This suggests that there was an error '
                   '(or preemption) between saving the numpy checkpoint and '
                   'the T5X checkpoint during the last save and that this is '
                   'a rerun. Moving that old checkpoint to %s',
                   numpy_dir,
                   backup_dir)
      gfile.rename(numpy_dir, backup_dir)
    gfile.rename(tmp_dir, numpy_dir, overwrite=True)
    logging.info('Saved Numpy Arrays for step %d to %s', step, numpy_dir)

  # TODO: See if the `state_transformation_fns` can replace our
  # checkpointer subclass to save numpy arrays.
  def save(self,
           train_state: train_state_lib.TrainState,
           state_transformation_fns: Sequence[
               checkpoints.SaveStateTransformationFn] = (),
           *,
           concurrent_gb: int = 128):
    self.save_numpy(train_state.params, train_state.step)
    super().save(
        train_state, state_transformation_fns, concurrent_gb=concurrent_gb)
