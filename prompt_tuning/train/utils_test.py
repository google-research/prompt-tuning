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

"""Tests for utils."""

import inspect
import operator
import os
from unittest import mock
from absl.testing import absltest
import flax
from flax import traverse_util
import numpy as np
from prompt_tuning.train import utils


def create_fake_parameters():
  qkv = 20
  p_length = 2
  embed_dim = 200
  layers = 4
  parameters = {}
  for i in range(layers):
    for proj in ('query', 'key', 'value', 'out'):
      parameters[f'encoder/layers_{i}/attention/'
                 f'{proj}/kernel'] = np.random.rand(qkv, qkv),
  parameters['encoder/prompt/prompt'] = np.random.rand(p_length, embed_dim)
  parameters = flax.core.freeze(traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in parameters.items()}))
  return parameters


class UtilsTest(absltest.TestCase):

  def test_match_any(self):
    regexes = ['.*', 'example']
    self.assertTrue(utils.match_any(regexes)('example', None))

  def test_match_any_closure(self):
    regexes = ['.*']
    self.assertTrue(callable(utils.match_any(regexes)))

  def test_match_any_full_mach(self):
    regexes = ['dogs']
    # This shouldn't match because we have extra text at the start.
    parameter = 'i/love/dogs'
    match_any = utils.match_any(regexes)
    self.assertFalse(match_any(parameter, None))

  def test_np_save(self):
    write_array = np.reshape(np.arange(20), (4, -1))
    test_file = self.create_tempfile('written_array.npy')
    test_file_name = test_file.full_path
    utils.np_save(test_file_name, write_array)
    read_array = np.load(test_file_name)
    np.testing.assert_allclose(read_array, write_array)

  def test_checkpointer_saves_prompt(self):
    parameters = create_fake_parameters()
    checkpoint_dir = self.create_tempdir().full_path
    step = 42
    time = 14
    golds = [
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_0.attention.key.kernel',
         parameters['encoder']['layers_0']['attention']['key']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_1.attention.query.kernel',
         parameters['encoder']['layers_1']['attention']['query']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_2.attention.value.kernel',
         parameters['encoder']['layers_2']['attention']['value']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_3.attention.out.kernel',
         parameters['encoder']['layers_3']['attention']['out']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.prompt.prompt', parameters['encoder']['prompt']['prompt'])]
    variable_paths = ['.*/prompt/.*',
                      '.*/layers_0/.*/key/.*',
                      '.*/layers_1/.*/query/.*',
                      '.*/layers_2/.*/value/.*',
                      '.*/layers_3/.*/out/.*']

    def mock_init(self, save_paths):
      self.checkpoints_dir = checkpoint_dir
      self.save_matcher = utils.match_any(save_paths)
      self._save_dtype = np.float32

    with mock.patch.object(utils.Checkpointer, '__init__', new=mock_init):
      checkpointer = utils.Checkpointer(variable_paths)

    with mock.patch.object(utils, 'np_save') as save_mock:
      with mock.patch.object(utils.time, 'time') as time_mock:
        time_mock.return_value = time
        checkpointer.save_numpy(parameters, step=step)

    self.assertLen(save_mock.call_args_list, len(golds))
    call_args = sorted((call_args[0] for call_args in save_mock.call_args_list),
                       key=operator.itemgetter(0))
    for (saved_path, saved_var), gold in zip(call_args, golds):
      self.assertEqual(saved_path, os.path.join(checkpoint_dir, gold[0]))
      np.testing.assert_allclose(saved_var,
                                 np.array(gold[1]).astype(np.float32))

  def test_checkpointer_with_existing_numpy_checkpoint(self):
    parameters = create_fake_parameters()
    checkpoint_dir = self.create_tempdir().full_path
    time = 14
    backup_time = 16
    step = 42
    golds = [
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_0.attention.key.kernel',
         parameters['encoder']['layers_0']['attention']['key']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_1.attention.query.kernel',
         parameters['encoder']['layers_1']['attention']['query']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_2.attention.value.kernel',
         parameters['encoder']['layers_2']['attention']['value']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.layers_3.attention.out.kernel',
         parameters['encoder']['layers_3']['attention']['out']['kernel']),
        (f'{checkpoint_dir}/numpy_checkpoints/checkpoint_{step}.tmp-{time}/'
         'encoder.prompt.prompt', parameters['encoder']['prompt']['prompt'])]
    variable_paths = ['.*/prompt/.*',
                      '.*/layers_0/.*/key/.*',
                      '.*/layers_1/.*/query/.*',
                      '.*/layers_2/.*/value/.*',
                      '.*/layers_3/.*/out/.*']

    def mock_init(self, save_paths):
      self.checkpoints_dir = checkpoint_dir
      self.save_matcher = utils.match_any(save_paths)
      self._save_dtype = np.float32

    with mock.patch.object(utils.Checkpointer, '__init__', new=mock_init):
      checkpointer = utils.Checkpointer(variable_paths)

    # Create a directory for an old checkpoint with this step, as if we are
    # saving after an error/preemption between the numpy checkpoint and the t5x
    # checkpoint.
    dir_already_there = os.path.join(checkpoint_dir,
                                     'numpy_checkpoints',
                                     f'checkpoint_{step}')
    self.create_tempfile(dir_already_there)

    with mock.patch.object(utils, 'np_save') as save_mock:
      with mock.patch.object(utils.time, 'time') as time_mock:

        def time_dispatch():
          """Create fake return values for our time patch.

          When we patch the time method, the patched version gets called by
          other modules like logging. This means we can't use the simple
          `.side_effect = (time, backup_time)` to return two values, one for
          the initial call to make a tmp dir and one to use when creating the
          backup. The number of these other calls seem to change as the number
          of sets change so we can't just use a longer sequence.

          Instead we check our call stack to see the context of the call.
          We skip 0, 1, and 2 because those calls are for '_time_dispatch',
          '_mock_call', and '__call__' which are mock calls and consistent.
          The 3rd frame gives us the calling function and lets us detect if
          we should return the backup time or not.

          Returns:
            The fake time.
          """
          fake_time = time
          def _time_dispatch():
            nonlocal fake_time
            if inspect.stack()[3].function == 'save_numpy':
              ret = fake_time
              fake_time = backup_time
              return ret
            else:
              return time
          return _time_dispatch

        time_mock.side_effect = time_dispatch()
        checkpointer.save_numpy(parameters, step=step)

    # Make sure the backup dir of the last checkpoint was created.
    backup_dir = f'{dir_already_there}.backup-{backup_time}'
    self.assertTrue(os.path.exists(backup_dir))

    # Check that the final checkpoint is still correct.
    self.assertLen(save_mock.call_args_list, len(golds))
    call_args = sorted((call_args[0] for call_args in save_mock.call_args_list),
                       key=operator.itemgetter(0))
    for (saved_path, saved_var), gold in zip(call_args, golds):
      self.assertEqual(saved_path, os.path.join(checkpoint_dir, gold[0]))
      np.testing.assert_allclose(saved_var,
                                 np.array(gold[1]).astype(np.float32))

  def test_checkpointer_saves_nothing(self):
    parameters = create_fake_parameters()
    checkpoint_dir = self.create_tempdir().full_path
    variable_paths = []
    step = 42

    def mock_init(self, save_paths):
      self.checkpoints_dir = checkpoint_dir
      self.save_matcher = utils.match_any(save_paths)
      self._save_dtype = np.float32

    with mock.patch.object(utils.Checkpointer, '__init__', new=mock_init):
      checkpointer = utils.Checkpointer(variable_paths)

    with mock.patch.object(utils, 'np_save') as save_mock:
      checkpointer.save_numpy(parameters, step=step)

    save_mock.assert_not_called()


if __name__ == '__main__':
  absltest.main()
