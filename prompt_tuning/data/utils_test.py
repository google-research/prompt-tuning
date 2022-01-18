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

from absl.testing import absltest
from prompt_tuning.data import utils


class UtilsTest(absltest.TestCase):

  def test_task_mapping(self):
    tasks = ['a', 'alias_a', 'alias_b', 'b', 'c', 'alias_b2']
    aliases = {
        'alias_a': 'a',
        'alias_b': 'b',
        'alias_b2': 'b'
    }
    gold_tasks = {
        'a': 0,
        'b': 1,
        'c': 2,
        'alias_a': 0,
        'alias_b': 1,
        'alias_b2': 1,
    }
    task_index = utils.task_mapping(tasks, aliases)
    self.assertEqual(task_index, gold_tasks)

  def test_task_mapping_raises_error(self):
    with self.assertRaises(ValueError):
      utils.task_mapping([], {'alias': 'missing_task'})


if __name__ == '__main__':
  absltest.main()
