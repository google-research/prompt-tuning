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

"""Utilities for defining tasks."""

from typing import Sequence, Mapping, Optional, Any, Dict


def identity(x: Any, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
  """Identity function used when a task doesn't have a postprocess function."""
  return x


def task_mapping(tasks: Sequence[str],
                 aliases: Optional[Mapping[str, str]]) -> Dict[str, int]:
  """Create a mapping from task name to index, sorted by task name.

  Args:
    tasks: The tasks that we are creating an index for. If any alias appears in
      these tasks it is removed before the sorting happens.
    aliases: Optional alternative names for a task, generally used for things
      like the SuperGLUE auxiliary tasks where things like the AX-b task is cast
      as the RTE task.

  Raises:
    ValueError if a alias is supposed to map to a task that was not provided.

  Returns:
    A mapping from task names to task indexes, where the tasks have been
    assigned based on the sorted task names. If any aliases are provided, they
    map to the same index as the task they alias.
  """
  # From the python 3.9.6 documentation for a set:
  # Return a new set or frozenset object whose elements are taken from iterable.
  # tasks now refers to a different set from the one possibly passed in so we
  # can do in-place operations like `-=` to it later.
  tasks = set(tasks)
  if aliases is None:
    aliases = {}
  # Remove any aliases from the list of tasks.
  tasks -= set(aliases.keys())
  task_index = {task: i for i, task in enumerate(sorted(tasks))}
  for alias, target in aliases.items():
    if target not in task_index:
      raise ValueError("You are trying to create a task alias from "
                       f"{alias}->{target} but {target} is not a provided "
                       "task.")
    task_index[alias] = task_index[target]
  return task_index


def remove_prefix(s: str, prefix: str) -> str:
  """Remove prefix from the beginning of the string if present."""
  if s.startswith(prefix):
    return s[len(prefix):]
  return s[:]


def remove_suffix(string: str, suffix: str) -> str:
  """Remove suffix from the end of the string if present."""
  if suffix and string.endswith(suffix):
    return string[:-len(suffix)]
  return string[:]
