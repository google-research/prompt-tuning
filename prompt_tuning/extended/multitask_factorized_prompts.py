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

"""Core modeling and utilities required for factorized prompts."""

from typing import Callable, Optional, Any, Iterable, Tuple
from flax import linen as nn
import jax.numpy as jnp
from flaxformer.components import embedding
from flaxformer.types import Array

Combine = Optional[Callable[[Array, Array, int, int], Array]]


def identity(x: Any) -> Any:
  """Return `x`, used as default function."""
  return x


def sentinel_to_task(x: Array, vocab_size: int) -> Array:
  r"""Convert the vocabulary index of tokens like <extra_id\d+> to 0 offsets."""
  return vocab_size - 1 - x


# ===== MultiTask Lang and Task-specific combination =====
def concat_langs_and_tasks(langs: Array, tasks: Array, lang_length: int,
                           task_length: int) -> Array:
  """Combine shared and task prompts via sequence length concatenation.

  Args:
    langs: [B, P_l * H] A batch of language specific prompts.
    tasks: [B, P_t * H] A batch of task specific prompts.
    lang_length: The length of the language prompt, P_l.
    task_length: The length of the task prompt, P_t.

  Returns:
    Task specific prompts concatenated to a shared prompt along the sequence
    dimension. [B, P_s + P_t, H]
  """
  embed_dim = langs.shape[1] // lang_length
  langs = jnp.reshape(langs, (-1, lang_length, embed_dim))
  tasks = jnp.reshape(tasks, (-1, task_length, embed_dim))
  return jnp.concatenate([langs, tasks], axis=1)


# TODO: Update these classes to use actual subclasses with
# overridden methods once that problem is fixed.
class MultiTaskFactorizedPrompt(nn.Module):
  """A module that produces a, combined, language and task specific prompt.

  Attributes:
    lang_prompt_length: The length of the language prompt, P_l.
    task_prompt_length: The length of the task prompt, P_t.
    num_langs: The number of languages that are included in our multi-task
      setup.
    num_tasks: The number of tasks that are included in our multi-task setup.
    to_lang_idx: A function that maps the first token of each element in the
      batch to a language index starting from zero.
    to_task_idx: A function that maps the second token of each element in the
      batch to a task index starting from zero.
    langs_init: An initializer for the language prompt variables.
    tasks_init: An initializer for the task prompt variables.
    combine_langs_and_tasks: A function that combines the (flattened) language
      and task specific prompts.
  """
  lang_prompt_length: int
  task_prompt_length: int
  num_langs: int
  num_tasks: int
  to_lang_idx: Callable[[Array], Array] = identity
  to_task_idx: Callable[[Array], Array] = identity
  langs_init: Callable[[Any, Iterable[int], Any],
                       Any] = nn.initializers.uniform()
  tasks_init: Callable[[Any, Iterable[int], Any],
                       Any] = nn.initializers.uniform()
  combine_langs_and_tasks: Combine = None
  lang_prompt_axis_names: Tuple[str, str] = ("langs", "prompt+embed")
  task_prompt_axis_names: Tuple[str, str] = ("tasks", "prompt+embed")

  def setup(self):
    if self.combine_langs_and_tasks is None:
      raise ValueError("`combine_langs_and_tasks` needs to be defined as a"
                       "function that combines arrays.")

  @nn.compact
  def __call__(self, x, x_embed):
    # Get the size of the (flattened) language/task prompts.
    embed_dim = x_embed.shape[-1]

    # Get the lang indices and convert them to offsets from zero.
    input_lang_idx = x[:, 0]
    lang_idx = self.to_lang_idx(input_lang_idx)  # pylint: disable=too-many-function-args

    # Lookup the language factors for the specific languages used.
    # We know that our functions are actually always callable, we raise an error
    # if they are None before getting here so this is a pylint false-positive.
    lang_prompts = embedding.Embed(
        self.num_langs,
        self.lang_prompt_length * embed_dim,
        embedding_init=self.langs_init,
        cast_input_dtype=jnp.int32,
        axes=self.lang_prompt_axis_names,
        name="lang_prompts")(
            lang_idx)

    # Get the task indices and convert them to offsets from zero.
    input_task_idx = x[:, 1]
    task_idx = self.to_task_idx(input_task_idx)  # pylint: disable=too-many-function-args

    # Lookup the task factors for the specific tasks used.
    # We know that our functions are actually always callable, we raise an error
    # if they are None before getting here so this is a pylint false-positive.
    task_prompts = embedding.Embed(
        self.num_tasks,
        self.task_prompt_length * embed_dim,
        embedding_init=self.tasks_init,
        cast_input_dtype=jnp.int32,
        axes=self.task_prompt_axis_names,
        name="task_prompts")(
            task_idx)

    # Combine the lang and task specific prompts.
    prompt = self.combine_langs_and_tasks(    # pylint: disable=not-callable
        lang_prompts,
        task_prompts,
        self.lang_prompt_length,
        self.task_prompt_length)
    assert prompt.shape[-1] == x_embed.shape[-1]
    return prompt


class ConcatMultiTaskFactorizedPrompt(MultiTaskFactorizedPrompt):
  """Create a MultiTask Prompt by concatenation along the sequence dimension.

  Attributes:
    combine_langs_and_tasks: A function that combines the (flattened) language
      and task specific prompts.
  """
  combine_langs_and_tasks: Combine = concat_langs_and_tasks
