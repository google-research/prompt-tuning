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

"""Training subclasses of our prompt that actually do the concatenation.

This also creates a unified API between single task prompts, where the prompt is
unbatched, and multi-task prompts, where the input needs to have the first token
removed.
"""

from typing import Callable
import flax.linen as nn
from prompt_tuning.train import prompts
from flaxformer.types import Array


class MultiTaskPrompt(nn.Module):
  """Generate a MultiTaskPrompt and concatenate it with the input.

  This is the training time version of prompting a model. Calling the injected
  `prompt` module will generate your prompt. This prompt should be batched. This
  module then slices off the task index from the input and concatenates the
  prompt. This can be used in conjunction with the `multitask=True` arguments
  for attention mask creation to do multi-task prompting without need multitask
  subclasses of various flaxformer modules.

  Attributes:
    prompt: The model that actually generates the batched prompt.
    combine: A function that combines the prompt and the embedded input.
  """
  prompt: nn.Module
  combine: Callable[[Array, Array, Array], Array] = prompts.prefix_prompt

  def __call__(self, x, x_embed):
    prompt = self.prompt(x, x_embed)
    # Remove the task index token
    x_embed = x_embed[:, 1:]
    # Pytype is throwing a false positive here, it probably thinks
    # `self.combine` is a method call that is giving a `self` parameter but it
    # is actually just a function so there are only 2 arguments, like the type
    # annotation says.
    return self.combine(prompt, x_embed, x)  # pylint: disable=too-many-function-args
