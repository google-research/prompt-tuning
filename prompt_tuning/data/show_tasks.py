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

r"""A quick tool to examine tasks.

Example usage:
See a list of all tasks this module defines:
python -m prompt_tuning.data.show_tasks

See the first example of the test split of the taskless boolq task:
python -m prompt_tuning.data.show_tasks \
  --task=taskless_super_glue_boolq_v102_examples \
  --split=test

"""
import importlib
from absl import app
from absl import flags
import byt5.tasks as byt5_tasks  # pylint: disable=unused-import
import multilingual_t5.tasks as mt5_tasks  # pylint: disable=unused-import
import seqio
import t5.data.mixtures as t5_mixtures  # pylint: disable=unused-import
import t5.data.tasks as t5_tasks  # pylint: disable=unused-import

OG_TASKS = frozenset(list(seqio.TaskRegistry._REGISTRY.keys()))  # pylint: disable=protected-access
OG_MIXTURES = frozenset(list(seqio.MixtureRegistry._REGISTRY.keys()))  # pylint: disable=protected-access

from prompt_tuning.data import tasks  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order

FLAGS = flags.FLAGS
flags.DEFINE_string("task", None, "The task you want to look at.")
flags.DEFINE_string("split", "validation",
                    "The split you want to look at in the task.")
flags.DEFINE_string("module",
                    None,
                    "An extra module containing tasks to import.")


def main(_):
  """Print all new tasks from the registry or a specific task."""
  if FLAGS.module is not None:
    importlib.import_module(FLAGS.module)
  if FLAGS.task is None:
    print("New tasks from `prompt_tuning.data.tasks`")
    for task in sorted(seqio.TaskRegistry._REGISTRY.keys()):  # pylint: disable=protected-access
      if task not in OG_TASKS:
        print(task)
    print("New mixtures from `prompt_tuning.data.tasks`")
    for mix in sorted(seqio.MixtureRegistry._REGISTRY.keys()):  # pylint: disable=protected-access
      if mix not in OG_MIXTURES:
        print(mix)
  else:
    task = seqio.get_mixture_or_task(FLAGS.task)
    dataset = task.get_dataset(None, split=FLAGS.split, shuffle=False)
    print(f"The first example from the {FLAGS.split} split of {FLAGS.task}:")
    batch = next(iter(dataset))
    for key, value in batch.items():
      if key.endswith("_pretokenized"):
        print(f"\t{key}:\n\t\t{value.numpy().decode('utf-8')}")
      else:
        print(f"\t{key}:\n\t\t{value.numpy()}")


if __name__ == "__main__":
  app.run(main)
