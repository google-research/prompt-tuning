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

"""Core modeling and utilities required for prompt tuning.

These are the canonical modules for creating prompts, they are designed to
return a prompt and be useable both during training where the prompt is part of
the module, and during inference (or client-server training) where it is passed
into the model.
"""

import re
from typing import Callable, Optional, Sequence, Tuple
from absl import logging
from flax import linen as nn
from flax import traverse_util
from flax.linen import partitioning
import jax
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import checkpoints
from tensorflow.io import gfile
from flaxformer.types import Array
from flaxformer.types import DType


Initializer = Callable[[Array, Sequence[int]], Array]


def expand_to_batch(x: Array, y: Array):
  """Expand unbatched `x` to the same batch size as `y`."""
  batch_size = y.shape[0]
  return jnp.tile(
      jnp.expand_dims(x, axis=0), [batch_size] + [1 for _ in x.shape])


# ===== Initialization =====
def np_load(path: str, restore_dtype: Optional[DType] = None) -> Array:
  """Load a numpy array.

  Note:
    This is basically just a wrapper around `np.load` that opens the file with
    gfile before hand so that we can open remote files.

  Args:
    path: Where the numpy `.npy` file lives.
    restore_dtype: What data type to cast the loaded array to.

  Returns:
    The numpy array read from the file.
  """
  with gfile.GFile(path, "rb") as f:
    arr = np.load(f)
  if restore_dtype is not None:
    arr = arr.astype(restore_dtype)
  return arr


def t5x_load(checkpoint_path: str,
             variable_path: str,
             restore_dtype: Optional[DType] = None) -> Array:
  """Load a numpy array from a t5x checkpoint.

  Args:
    checkpoint_path: The path to the checkpoint file in the t5x checkpoint
      directory.
    variable_path: The path to the variable inside the checkpoint. Use `/` for
      nesting. Leading `target/` is optional.
    restore_dtype: What data type to cast the loaded array to.

  Returns:
    The numpy array read from the checkpoint.
  """
  ckpt = checkpoints.load_t5x_checkpoint(checkpoint_path,
                                         restore_dtype=restore_dtype,
                                         use_gda=False,
                                         lazy_parameters=True)
  ckpt = {"/".join(k): v for k, v in traverse_util.flatten_dict(ckpt).items()}
  variable_path = re.sub(r"^(target)?/", "", variable_path)
  variable_path = f"target/{variable_path}"
  return ckpt[variable_path].get()


def from_array(prompt: Array) -> Initializer:
  """Initialize a prompt with a value passed in.

  This can be used to load a prompt from a file. Example Gin Config:

  ```
  prompts.Prompt:
    length = 20
    prompt_init = @prompts.from_array()

  prompts.from_array.prompt = @prompts.np_load()
  prompts.np_load.path = "/path/to/my/prompt"
  ```

  Args:
    prompt: [P, H] The array to use to initialize the prompt.

  Returns:
    A closure over the array that can be used as a flax initializer.
  """

  def initialize_from_file(unused_prng_key: Array,
                           shape: Sequence[int]) -> Array:
    """Return the loaded prompt as an initial value.

    Args:
      unused_prng_key: A PRNGKey that we don't use.
      shape: The shape we expect the prompt to be.

    Raises:
      ValueError if the loaded prompt doesn't match the provided shape.

    Returns:
      The prompt as a jax array. [P, H]
    """
    del unused_prng_key
    if prompt.shape != shape:
      raise ValueError(f"The prompt loaded from disk has shape: {prompt.shape}"
                       f"while the configs says the shape should be: {shape}")
    return jnp.array(prompt)

  return initialize_from_file


def from_sample_of_embeddings(
    embeddings: Array,
    population_size: Optional[int] = None,
) -> Callable[[Array, Sequence[int]], Array]:
  """Initialize by drawing vectors from the embedding table.

  Note:
    If not provided, the population size used is the full possibility of the
    vector space.

  Args:
    embeddings: [V, H] The embeddings to draw vectors from.
    population_size: Limit the drawing to the first `population_size` vectors.

  Returns:
    A closure over the embedding table that can be used as a flax initializer.
  """
  if population_size is None:
    population_size = embeddings.shape[0]
  if population_size <= 0:
    raise ValueError(f"Cannot sample from a population less than zero. Got "
                     f"{population_size}")
  if population_size > embeddings.shape[0]:
    logging.warning(
        "The requested `population_size` (%d) is larger than the "
        "total available embeddings (%d). Setting "
        "`population_size` to the embedding size.", population_size,
        embeddings.shape[0])
    population_size = embeddings.shape[0]
  # Because our sampling is done with jax (so that we can make use of the rng
  # key), we need our embeddings to be in jax, otherwise we get errors because
  # the indices will be a jax tracer and it fails when it is converted to numpy
  # to lookup values in a number array. This call pins the embeddings to cpu so
  # we don't waste TPU memory carrying it around.
  embeddings = jax.device_put(embeddings, jax.devices("cpu")[0])

  def initialize_from_embedding_sample(rng: Array,
                                       shape: Sequence[int]) -> Array:
    """Sample from the embedding table, without replacement.

    Note:
      If the number of prompt tokens requested is larger than the total number
      of vectors we are drawing from (`population_size`) we do sampling with
      replacement.

    Args:
      rng: The rng seed used in our sampling.
      shape: The shape of the prompt variable. shape[0] tells us how many
        vectors to sample.

    Raises:
      ValueError if the number of features in the embedding table do not match
      the number of features in the prompt.

    Returns:
      A sample of the embedding table as a jax array. [P, H]
    """
    if embeddings.shape[-1] != shape[-1]:
      raise ValueError(
          "Shape mismatch between the number of features in the "
          f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
          f"{shape[-1]}.")
    replace = False
    if shape[0] > population_size:
      logging.warning(
          "Prompt Length: %d is larger than the number of vectors "
          "to draw from: %d. Switching to draws with replacement.", shape[0],
          population_size)
      replace = True
    index = jax.random.choice(rng, population_size, (shape[0],), replace)
    prompt = embeddings[index]
    return jnp.array(prompt)

  return initialize_from_embedding_sample


# TODO: Add parameter that lets you decide if you want to aggregate
# a label that is broken into multiple subwords into a single word.
# TODO: Add parameter that lets you pass in an aggregation
# function.
# TODO: Create an initializer where the texts are read from a file
# and passed to this initializer.
def from_embedded_list(
    embeddings: Array,
    vocab: seqio.SentencePieceVocabulary,
    texts: Sequence[str],
    initializer: Optional[Callable[[Array, Sequence[int]], Array]] = None
) -> Callable[[Array, Sequence[int]], Array]:
  """Initialize with the embedded values of a list of words.

  This can be used in conjunction with the `sample_from_embeddings` initializer
  to replicate the "Class Label" initialization from the
  [prompt tuning paper](https://arxiv.org/abs/2104.08691). Example Gin config.

  ```
  prompts.Prompt:
    length = 20
    prompt_init = @prompts.from_embedded_list()

  prompts.from_embedded_list:
    embeddings = @prompts.np_load()
    vocab = @seqio.SentencePieceVocabulary()
    texts = ["True", "False"]
    initializer = @prompts.from_sample_of_embeddings()

  prompts.from_sample_of_embeddings:
    embeddings = @prompts.np_load()
    population_size = 5000

  prompts.np_load.path = "/path/to/my/embeddings"
  seqio.SentencePieceVocabulary.sentencepiece_model_file = "/path/to/my/vocab"
  ```

  This does require a double load of the embedding table but there isn't a great
  way to avoid that because gin doesn't cache the results of calls, if it did
  we wouldn't have to reread the embedding table when the second initializer
  calls np_load to get its initial value.

  Args:
    embeddings: [V, H] The embedding table we are looking up words in.
    vocab: The vocabulary to convert strings to integers.
    texts: The list of words to embed for initialization.
    initializer: An initializer used to repopulate the prompt in case the
      provided list of strings is shorter than the prompt length.

  Returns:
    A closure over the embeddings and vocab that returns the initialized prompt.
  """

  def initialize_from_embedded_list(rng: Array, shape: Sequence[int]) -> Array:
    """Create a prompt by embedding given words.

    Note:
      In the case that a provided work is broken into multiple pieces by the
      vocabulary, the mean of the resulting embedding vectors is used as the
      initialization.

    Args:
      rng: The jax rng that is passed to the sub-initializer.
      shape: The shape of the prompt we are making. `shape[0]` gives us the
        length of the prompt.

    Raises:
      ValueError if the number of features in the embedding table don't match
      the number of features in the prompt.

    Returns:
      The prompt with `prompt[i]` initialized with the value of
      `embeddings[vocab.encode(texts[i])]`. [P, H]
    """
    if embeddings.shape[-1] != shape[-1]:
      raise ValueError(
          "Shape mismatch between the number of features in the "
          f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
          f"{shape[-1]}.")
    if initializer is None:
      prompt = nn.initializers.uniform()(rng, shape)
    else:
      prompt = initializer(rng, shape)

    for i, text in enumerate(texts):
      if i >= len(prompt):
        logging.warning(
            "Ran out of prompt positions before initializing with "
            "all the provided text. %s has been used for "
            "initialization and %s will be skipped.", texts[:i], text[i:])
        break
      encoded_text = vocab.encode(text)
      embedded_text = embeddings[encoded_text]
      embedded_text = jnp.mean(embedded_text, axis=0)
      prompt = prompt.at[i].set(embedded_text)

    return jnp.array(prompt)

  return initialize_from_embedded_list


def from_embedded_string(
    embeddings: Array,
    vocab: seqio.SentencePieceVocabulary,
    text: str,
    initializer: Optional[Callable[[Array, Sequence[int]], Array]] = None
) -> Callable[[Array, Sequence[int]], Array]:
  """Initialize with the embedded values of a string.

  Args:
    embeddings: [V, H] The embedding table we are looking up words in.
    vocab: The vocabulary to convert strings to integers.
    text: The string to embed for initialization.
    initializer: An initializer used to repopulate the prompt in case the
      provided list of strings is shorter than the prompt length.

  Returns:
    A closure over the embeddings and vocab that returns the initialized prompt.
  """

  def initialize_from_embedded_string(rng: Array,
                                      shape: Sequence[int]) -> Array:
    """Create a prompt by embedding given words.

    Args:
      rng: The jax rng that is passed to the sub-initializer.
      shape: The shape of the prompt we are making. `shape[0]` gives us the
        length of the prompt.

    Raises:
      ValueError if the number of features in the embedding table don't match
      the number of features in the prompt.

    Returns:
      The prompt with `prompt[i]` initialized with the value of
      `embeddings[vocab.encode(text)[i]]`. [P, H]
    """
    if embeddings.shape[-1] != shape[-1]:
      raise ValueError(
          "Shape mismatch between the number of features in the "
          f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
          f"{shape[-1]}.")
    if initializer is None:
      prompt = nn.initializers.uniform()(rng, shape)
    else:
      prompt = initializer(rng, shape)

    segmented_text = vocab.tokenizer.EncodeAsPieces(text)
    segmented_ids = vocab.encode(text)
    if len(segmented_ids) > len(prompt):
      logging.warning(
          "Ran out of prompt positions before initializing with "
          "all the provided text. %s has been used for "
          "initialization and %s will be skipped.",
          segmented_text[:len(prompt)],
          segmented_text[len(prompt)])
    embedded_text = embeddings[segmented_ids[:len(prompt)]]
    prompt = prompt.at[list(range(len(embedded_text))), :].set(embedded_text)

    return jnp.array(prompt)

  return initialize_from_embedded_string


class Prompt(nn.Module):
  """A module that produces a learnable prompt.

  Attributes:
    length: The length of the prompt, P.
    prompt_init: An initializer function for the variable.
    axis_names: Logical names for the parameter axes. Note: We use
      "prompt_embed" as the second dimension so that the prompt is always
      replicated, even when using 2-way parameter partitioning when the "embed"
      dimension would get partitioned. This makes it possible to save the prompt
      as a numpy file. If the prompt needs to be partitioned, one can change the
      second dimension to "embed", but the prompt variable will need to be
      managed by the t5x checkpointing utilities (i.e. the numpy checkpoint will
      not be the full prompt and you will need to save multiple t5x checkpoints)
      and `prompt_tuning.scripts.extract_variable` to create a numpy checkpoint.
    dtype: The dtype of the activations for this module.
  """
  length: int
  prompt_init: Initializer = nn.initializers.uniform()
  axis_names: Tuple[str, str] = ("prompt", "prompt_embed")
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x, x_embed):
    """Get the prompt variable.

    Args:
      x: [B, T] The sequence of input tokens.
      x_embed: [B, T, H] The sequence of embedded input tokens.

    Returns:
      The prompt variable. [P, H]
    """
    embed_dim = x_embed.shape[-1]
    prompt = partitioning.param_with_axes(
        "prompt",
        self.prompt_init,
        (self.length, embed_dim),
        axes=self.axis_names)
    prompt = prompt.astype(self.dtype)
    return prompt


class InferenceOnlyPrompt(nn.Module):
  """An inference only prompt that reads from a file.

  This is a Prompt that sidesteps some memory issues when loading a prompt for
  inference. Instead of the prompt being an actual parameter of the model, it
  is loaded with a constant that is not updated over time.

  When using this module `fallback_to_scratch` should be set to `false`. This
  module is generally used in conjunction with `prompts/from_file.gin`.

  Example configuration:
  ```
  prompts.Prompt.prompt_init = @prompt_init/prompts.from_array()
  prompt_init/prompts.from_array.prompt = @prompt_init/prompts.np_load()
  prompt_init/prompts.np_load.path = "/path/to/your/prompt/file.npy"
  ```

  Attributes:
    length: The length of the prompt.
    embed_dim: The size of the embeddings (number of features in the prompt).
    prompt_init: How to initialize the prompt. Generally this will be used to
      load a pre-trained prompt from disk.
    dtype: What dtype the prompt should be.
  """
  length: int
  embed_dim: int
  prompt_init: Initializer
  dtype: DType = jnp.float32

  def setup(self):
    self.prompt = jnp.array(
        self.prompt_init(jax.random.PRNGKey(0), (self.length, self.embed_dim)))

  def __call__(self, x, x_embed):
    return self.prompt.astype(self.dtype)
