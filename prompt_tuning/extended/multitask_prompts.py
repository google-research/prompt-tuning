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

"""Core modeling and utilities required for prompt tuning.

These are non-standard prompts that are designed around multi-tasking where the
prompt is split into a variable that is shared across all tasks and variables
that are specific to some task. This code assumes that the first token in each
example is an index that specifies which task is being done. They are designed
to return a prompt and be useable both during training where the prompt is part
of the module, and during inference (or client-server training) where it is
passed into the model.

Note:
  There are several multitask options to pick from, they mostly differ on how
  they combine the shared prompt and the task specific prompts and how the size
  for the task specific prompt is calculated. They would make more sense as
  subclasses where there are methods like `combine_shared_and_tasks` or
  `get_task_size` that are overridden in subclasses. We were not able to get
  this to work. When running inside of T5X on TPU, we couldn't stop JAX from
  outputting traced values for methods like `get_shared_features`, despite it
  being based on shape information only. This caused errors when we used the
  result in the prompt shape.

  TODO: Create a minimum reproducible example in colab, file a bug,
  and link that bug here.

  We solved this by turning my methods into explicit functions that are class
  attributes (with defaults). When these functions are called, the module's
  `self` is explicitly passed in. This replicates having methods on subclasses
  but avoids the JAX error and allows us to use the results of computation for
  the parameter sizes.
"""

from typing import Callable, Optional, Any, Iterable, Tuple
from flax import linen as nn
from flax.linen import partitioning
import jax.numpy as jnp
from prompt_tuning.prompts import expand_to_batch
from prompt_tuning.prompts import Initializer
from flaxformer.components import embedding
from flaxformer.types import Array
from flaxformer.types import DType


GetFeatures = Optional[Callable[[nn.Module, int], int]]
GetTaskSize = Optional[Callable[[nn.Module, int, int], int]]
Combine = Optional[Callable[[Array, Array], Array]]


def identity(x: Any) -> Any:
  """Return `x`, used as default function."""
  return x


def sentinel_to_task(x: Array, vocab_size: int) -> Array:
  r"""Convert the vocabulary index of tokens like <extra_id\d+> to 0 offsets."""
  return vocab_size - 1 - x


# ===== MultiTask Shared and Task-specific combination =====
def add_shared_and_tasks(shared: Array, tasks: Array) -> Array:
  """Combine shared and task prompts via addition.

  Args:
    shared: [P, H] A prompt that is shared across tasks (unbatched).
    tasks: [B, P * H] A batch of task specific prompts.

  Returns:
    Tasks specific prompts that have been combined with the shared prompt.
    [B, P, H]
  """
  length, size = shared.shape
  return jnp.expand_dims(
      shared, axis=0) + jnp.reshape(tasks, (-1, length, size))


def concat_shared_and_tasks(shared: Array, tasks: Array) -> Array:
  """Combine shared and task prompts via sequence length concatenation.

  Args:
    shared: [P_s, H] A prompt that is shared across tasks (unbatched).
    tasks: [B, P_t * H] A batch of task specific prompts.

  Returns:
    Task specific prompts concatenated to a shared prompt along the sequence
    dimension. [B, P_s + P_t, H]
  """
  _, size = shared.shape
  task_length = tasks.shape[1] // size
  tasks = jnp.reshape(tasks, (-1, task_length, size))
  shared = expand_to_batch(shared, tasks)
  return jnp.concatenate([shared, tasks], axis=1)


def concat_features_shared_and_tasks(shared: Array, tasks: Array) -> Array:
  """Combine shared and task prompts via feature vector concatenation.

  Args:
    shared: [P, H_s] A prompt that is shared across tasks (unbatched).
    tasks: [B, P * H_t] A batch of task specific prompts.

  Returns:
    Task specific prompts concatenated to a shared prompts along the feature
    dimension. [B, P, H_s + H_t] s.t. H_s + H_t = H
  """
  length, _ = shared.shape
  task_size = tasks.shape[1] // length
  tasks = jnp.reshape(tasks, (-1, length, task_size))
  shared = expand_to_batch(shared, tasks)
  return jnp.concatenate([shared, tasks], axis=2)


def project_shared_with_tasks(shared: Array, tasks: Array) -> Array:
  """Project a shared prompt to a task specific prompt.

  This is a series of oddly shaped tensor manipulations but it has the same
  result looping over each position in the shared prompt and projecting it by
  the kernel and bias represented by each batch element. Essentially this is the
  same result as:

  ```
  projected_prompt = [B, P, H]
  for i in shared.shape[0]:  # Positions in the shared prompt
    for j in tasks.shape[0]:  # Each batch element is a task specific transform
      projected_prompt[j, i, :] = project(shared[i, :], tasks[j, :])
  ```


  Args:
    shared: [P, H_s] A prompt that is shared across tasks (unbatched).
    tasks: [B, H * H_s + H] A batch of task specific variables that include the
      parameters for the kernel and the bias of an affine transformation.

  Returns:
    Tasks specific prompts by projecting the shared prompt with different task
    weights. [B, P, H]
  """
  length, size = shared.shape
  # The task vector is a flattened kernel + a bias so it shape is
  # tasks.shape = (kernel_in * kernel_out) + kernel_out
  # tasks.shape = (kernel_in + 1) * kernel_out
  # tasks.shape / (kernel_in + 1) = kernel_out
  # and the task_size here is the output shape.
  task_size = tasks.shape[1] // (size + 1)
  kernel = jnp.reshape(tasks[:, :task_size * size], (-1, size))  # [B * H_s, H]
  bias = tasks[:, task_size * size:]  # [B, H]
  # [P, H_s] @ [H_s, B * H] -> [P, B * H]
  projection = shared @ jnp.transpose(kernel)
  projection = jnp.reshape(projection, (length, -1, task_size))  # [P, B, H]
  with_bias = projection + bias  # [P, B, H]
  return jnp.transpose(with_bias, (1, 0, 2))  # [B, P, H]


# ===== MultiTask Shape Inference =====
def input_features(self, x_embed_features: int) -> int:  # pylint: disable=unused-argument
  """Return the embedding features.

  Used as the default where the task specific prompt features are expected to
  match the size of the embeddings features.

  Note:
    This function is basically a method that is set as a class attribute to
    avoid issues of JAX returning tracer values for this function.

  Args:
    self: The Prompt Module.
    x_embed_features: The number of features in the embedding, H.

  Returns:
    The number of features in the shared prompt.
  """
  return x_embed_features


def shared_features_attr_or_input_features(self, x_embed_features: int) -> int:
  """Use class attribute for the embedding features, back off to matching.

  Note:
    This function is basically a method that is set as a class attribute to
    avoid issues of JAX returning tracer values for this function.

    This assumes there is a `shared_features`, H_s, attribute on your module.

  Args:
    self: The Prompt Module.
    x_embed_features: The number of features in the embedding, H.

  Returns:
    The number of features in the shared prompt.
  """
  return x_embed_features if self.shared_features is None else self.shared_features


def match_shared(self, shared_length: int, shared_features: int) -> int:  # pylint: disable=unused-argument
  """Calculate the size of the task variable to match the shared variable."""
  return shared_length * shared_features


def task_length_attr_or_match_shared(self, shared_length: int,
                                     shared_features: int) -> int:
  """Calculate (flattened) task variable size with a class attribute for length.

  Note:
    This function is basically a method that is set as a class attribute to
    avoid issues of JAX returning tracer values for this function.

    This assumes there is a `task_length`, P_t, attribute on your module.

  Args:
    self: The Prompt Module.
    shared_length: The number tokens in the prompt, P_s.
    shared_features: The number of features in each timestep of the prompt, H_s.

  Returns:
    The number of elements in a single task specific variable.
  """
  task_length = shared_length if self.task_length is None else self.task_length
  return task_length * shared_features


def task_features_or_match_shared(self, shared_length: int,
                                  shared_features: int) -> int:
  """Calculate (flattened) task variable size with a class attribute for feats.

  Note:
    This function is basically a method that is set as a class attribute to
    avoid issues of JAX returning tracer values for this function.

    This assumes there is a `task_features`, H_t, attribute on your module.

  Args:
    self: The Prompt Module.
    shared_length: The number tokens in the prompt, P_s.
    shared_features: The number of features in each timestep of the prompt, H_s.

  Returns:
    The number of elements in a single task specific variable.
  """
  task_features = shared_features
  if self.task_features is not None:
    task_features = self.task_features
  return shared_length * task_features


def projection_size(self, shared_length: int, shared_features: int) -> int:  # pylint: disable=unused-argument
  """Calculate the (flattened) task variable size when used as a projection.

  Note:
    This function is basically a method that is set as a class attribute to
    avoid issues of JAX returning tracer values for this function.

    This assumes there is a `task_features`, H_s, attribute on your module.

  Args:
    self: The Prompt Module.
    shared_length: The number tokens in the prompt, P_s.
    shared_features: The number of features in each timestep of the prompt, H_s.

  Returns:
    The number of elements in a single task specific variable.
  """
  task_features = shared_features
  if self.task_features is not None:
    task_features = self.task_features
  kernel = shared_features * task_features
  bias = task_features
  return kernel + bias


# TODO: Update these classes to use actual subclasses with
# overridden methods once that problem is fixed.
class MultiTaskPrompt(nn.Module):
  """A module that produces a, combined, shared and tasks specific prompt.

  Attributes:
    length: The length of the shared prompt, P_s.
    num_tasks: The number of tasks that are including in our multi-task setup.
    to_task_idx: A function that maps the first token of each element in the
      batch to a task index starting from zero.
    shared_init: An initializer for the shared prompt variables.
    task_init: An initializer for the task prompts variables.
    get_shared_features: A function (pretending to be a method) that calculates
      the number of features in the shared prompt.
    get_task_size: A function (pretending to be a method) that calculates the
      number of values in a single (flattened) task specific prompt.
    combine_shared_and_tasks: A function that combines the shared and
      (flattened) task specific prompts.
    shared_prompt_axis_names: Logical names for the axes of the shared prompt
      parameter.
    individual_prompt_axis_names: Logical names for the axes of the per-task
      prompt parameters.
    dype: The dtype of the activations of this module.
  """
  length: int
  num_tasks: int
  to_task_idx: Callable[[Array], Array] = identity
  shared_init: Initializer = nn.initializers.uniform()
  tasks_init: Callable[[Any, Iterable[int], Any],  # pytype: disable=annotation-type-mismatch  # jax-types
                       Any] = nn.initializers.uniform()
  get_shared_features: GetFeatures = input_features
  get_task_size: GetTaskSize = None
  combine_shared_and_tasks: Combine = None
  shared_prompt_axis_names: Tuple[str, str] = ("prompt", "embed")
  individual_prompt_axis_names: Tuple[str, str] = ("tasks", "prompt+embed")
  dtype: DType = jnp.float32

  def setup(self):
    if self.combine_shared_and_tasks is None:
      raise ValueError("`combine_shared_and_tasks` needs to be defined as a"
                       "function that combines arrays.")
    if self.get_task_size is None:
      raise ValueError("`get_task_size` needs to be defined as a function to "
                       "calculate the (flattened) size of a single task "
                       "specific parameter. The function should take the "
                       "object instance as the first parameter.")
    if self.get_shared_features is None:
      raise ValueError("`get_shared_features` needs to be defined as a "
                       "function to calculate the number of features in a "
                       "single timestep of the shared prompt. The function "
                       "should take the object instances as the first "
                       "parameter.")

  @nn.compact
  def __call__(self, x, x_embed):
    # Get the size of the shared prompt and initialize it.
    shared_features = self.get_shared_features(self, x_embed.shape[-1])  # pylint: disable=too-many-function-args
    shared_prompt = partitioning.param_with_axes(
        "shared_prompt",
        self.shared_init,
        (self.length, shared_features),
        axes=self.shared_prompt_axis_names)
    shared_prompt = shared_prompt.astype(self.dtype)

    # Get the task indices and convert them to offsets from zero.
    input_task_idx = x[:, 0]
    task_idx = self.to_task_idx(input_task_idx)  # pylint: disable=too-many-function-args

    # Lookup the task prompts for the specific tasks used.
    # We know that our functions are actually always callable, we raise an error
    # if they are None before getting here so this is a pylint false-positive.
    task_size = self.get_task_size(self, self.length, shared_features)  # pylint: disable=not-callable
    task_prompts = embedding.Embed(
        self.num_tasks,
        task_size,
        embedding_init=self.tasks_init,
        cast_input_dtype=jnp.int32,
        axes=self.individual_prompt_axis_names,
        name="task_prompts")(
            task_idx,
            input_axis_names=("batch",))

    # Combine the shared and task specific prompts.
    prompt = self.combine_shared_and_tasks(shared_prompt, task_prompts)  # pylint: disable=not-callable
    assert prompt.shape[-1] == x_embed.shape[-1]
    return prompt


class AddMultiTaskPrompt(MultiTaskPrompt):
  """Create a MultiTask Prompt by summing the shared and task prompts.

  Attributes:
    get_task_size: A function (pretending to be a method) that calculates the
      number of values in a single (flattened) task specific prompt.
    combine_shared_and_tasks: A function that combines the shared and
      (flattened) task specific prompts.
  """
  get_task_size: GetTaskSize = match_shared
  combine_shared_and_tasks: Combine = add_shared_and_tasks


class ConcatMultiTaskPrompt(MultiTaskPrompt):
  """Create a MultiTask Prompt by concatenation along the sequence dimension.

  Attributes:
    task_length: The length of the task specific prompt. Defaults to the length
      of the shared prompt.
    get_shared_features: A function (pretending to be a method) that calculates
      the number of features in the shared prompt.
    get_task_size: A function (pretending to be a method) that calculates the
      number of values in a single (flattened) task specific prompt.
    combine_shared_and_tasks: A function that combines the shared and
      (flattened) task specific prompts.
  """
  task_length: Optional[int] = None
  get_shared_features: GetFeatures = input_features
  get_task_size: GetTaskSize = task_length_attr_or_match_shared
  combine_shared_and_tasks: Combine = concat_shared_and_tasks


class ConcatFeaturesMultiTaskPrompt(MultiTaskPrompt):
  """Create a MultiTask Prompt by concatenation only the feature dimension.

  Attributes:
    task_features: The number of features in the task specific prompt.
    shared_features: The number of features in the shared prompt.
    get_shared_features: A function (pretending to be a method) that calculates
      the number of features in the shared prompt.
    get_task_size: A function (pretending to be a method) that calculates the
      number of values in a single (flattened) task specific prompt.
    combine_shared_and_tasks: A function that combines the shared and
      (flattened) task specific prompts.
  """
  task_features: Optional[int] = None
  shared_features: Optional[int] = None
  get_shared_features: GetFeatures = shared_features_attr_or_input_features
  get_task_size: GetTaskSize = task_features_or_match_shared
  combine_shared_and_tasks: Combine = concat_features_shared_and_tasks


class ProjectMultiTaskPrompt(MultiTaskPrompt):
  """Create a MultiTask Prompt by projecting with an offing transformation.

  Attributes:
    task_features: The number of features as the output of the transformation.
    shared_features: The number of features as the input of the transformation.
    get_shared_features: A function (pretending to be a method) that calculates
      the number of features in the shared prompt.
    get_task_size: A function (pretending to be a method) that calculates the
      number of values in a single (flattened) task specific prompt.
    combine_shared_and_tasks: A function that combines the shared and
      (flattened) task specific prompts.
  """
  task_features: Optional[int] = None
  shared_features: Optional[int] = None
  get_shared_features: GetFeatures = shared_features_attr_or_input_features
  get_task_size: GetTaskSize = projection_size
  combine_shared_and_tasks: Combine = project_shared_with_tasks
