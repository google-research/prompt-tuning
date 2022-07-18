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

"""Run all the recycling and generate cli arguments.

Configuration format:
{
  "root_dir": str  # The root dir to save everything into.
  "dataset": str  # What dataset is this all for?
  "task_name": str  # What is the name of the seqio task for collecting metrics.
  "clobber": bool  # Should we re-fit and overwrite saved recyclers?
  "steps": Optional[List[int]]  # The steps to recycle, None defaults to all.
  "combination_skips": Optional[List[Tuple[str, str]]]  # Source Target tuples to skip when recycling.  # pylint: disable=line-too-long
  "load_embeddings": {  # Methods and parameters on how to load embeddings.
    "${load_method}": {
      "${load_arg}": "arg_value"
    },
    ...
  },
  "recycling_methods": {  # Methods and parameters for learning recyclers.
    "${recycling_method_name}": {
      "${object_method} (__init__ for example)": {
        "${method_arg}": "arg_value"
      },
      ...
    },
    ...
  },
  "pretrained": {
    "${seed}": str,  # Where the pre-trained model for this seed lives.
    ...
  },
  "prompts": {
    "${seed}": {
      "${init}": {
        "${run}": str  # The path to the model_dir for training this prompt.,
        ...
      },
      ...
    },
    ...
  }
}
"""

import itertools
import json
import os
import re
from typing import List
from absl import app
from absl import flags
from absl import logging
import flax
from flax import traverse_util
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from prompt_tuning import prompts
from prompt_tuning.recycling import utils
import seqio
from t5x import checkpoints
import tensorflow as tf
from tensorflow.io import gfile

from flaxformer.types import Array


DEFAULT_CONFIG = "./prompt_tuning/recycling/configs/sst2.json"

_CONFIG_FILE = flags.DEFINE_string(
    "config",
    DEFAULT_CONFIG,
    "Configuration file with all the settings and information.")
_RECYCLE = flags.DEFINE_boolean(
    "recycle", True, "Should we actually do the recycling.")


def _make_dir(path: str) -> bool:
  """Makes a directory and returns true if it doesn't exist."""
  if not gfile.exists(path):
    gfile.makedirs(path)
    return True
  return False


def default_load_embeddings(
    path: str, word_list_path: str, num_words: int, word_offset: int) -> Array:
  """Load the first num_words english embeddings (skipping offset) from path."""
  embeddings = utils.load_english_only_embedding(path, word_list_path)
  selection = jnp.arange(word_offset, num_words + word_offset)
  embeddings = embeddings[selection]
  assert len(embeddings) == num_words
  return embeddings


def load_embeddings_from_string(
    path: str, vocab_file: str, variable_path: str, string: str) -> Array:
  """Load embeddings from path and return the embeddings for string."""
  ckpt = checkpoints.load_t5x_checkpoint(path, lazy_parameters=True)
  flat_ckpt = traverse_util.flatten_dict(ckpt, sep="/")
  variable_path = re.sub(r"^(target)?/?", "", variable_path)
  variable_path = f"target/{variable_path}"
  embedding_table = flat_ckpt[variable_path].get()
  vocab = seqio.vocabularies.SentencePieceVocabulary(vocab_file)
  encoded_string = vocab.encode(string)
  return embedding_table[encoded_string]


def load_embeddings(path: str, method: str, config) -> Array:
  """Load embeddings from path, dispatched on the method."""
  args = config["load_embeddings"][method]
  if method == "default":
    fn = default_load_embeddings
  elif method == "from_string":
    fn = load_embeddings_from_string
  else:
    raise ValueError(f"Couldn't understand string loading method {method}")
  return fn(path, **args)


class Recycler:
  """Recycle a prompt for one model to use on another."""

  def __call__(self, inputs: Array) -> Array:
    """Recycler `inputs`."""
    return inputs

  def loss_fn(self, inputs: Array, targets: Array) -> float:
    """Loss (L_2) between our recycling and the target."""
    pred = self.__call__(inputs)
    loss_value = optax.l2_loss(pred, targets)
    return loss_value.mean()

  def fit(self, inputs: Array, targets: Array, **kwargs) -> "Recycler":
    """Learn a transformation from the `inputs` to the `targets`."""
    del inputs
    del targets
    del kwargs
    return self

  def save(self, path: str):
    """Save recycler parameters to `path`."""
    del path
    pass

  def load(self, path: str) -> "Recycler":
    """Load recycler parameters from `path`."""
    del path
    return self


class TFLstSqRecycler(Recycler):
  """Recycle a prompt by fitting a linear projection with LstSqs."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Create /something/ for self.projection or pylint complains.
    self.projection = tf.zeros((1, 1))

  def __call__(self, inputs: Array) -> Array:
    """Recycle the prompt with a single projection."""
    # [P, E] @ [E, E_2] = [P, E_2]
    return inputs @ self.projection

  def fit(self, inputs: Array, targets: Array, **kwargs) -> Recycler:
    """Fit the projection with LstSqs."""
    # LstSq, find x st. a @ x = b
    # a = inputs [V, E]
    # b = targets [V, E_2]
    # x = self.projection [E, E_2]
    self.projection = tf.linalg.lstsq(inputs, targets).numpy()
    logging.info("Loss: %f", self.loss_fn(inputs, targets))
    return self

  def save(self, path: str):
    """Save the recycler to `path`."""
    np_save(path, self.projection)

  def load(self, path: str) -> Recycler:
    """Load projection from path. No validation of method checkpoint is done."""
    self.projection = prompts.np_load(path)
    return self


class ReLUMLP(nn.Module):
  """2 layer expand-and-contract MLP with ReLU activations.

  Attributes:
    hidden: The number of units in the hidden layer.
    output: The number of units in the output layer.
  """
  hidden: int
  output: int

  @nn.compact
  def __call__(self, x):
    """Run W_2@relu(W_1@x + b_1) + b_2."""
    x = nn.Dense(self.hidden)(x)
    x = nn.relu(x)
    return nn.Dense(self.output)(x)


class JaxRecycler(Recycler):
  """Recycle a prompt by fitting a small MLP using Jax."""

  def __init__(self, hidden_scale: int, output: int, input_: int):
    """Create the MLP and initialize parameters."""
    self.recycler = ReLUMLP(hidden_scale * output, output)
    self.params = self.recycler.init(jax.random.PRNGKey(0),
                                     jnp.ones((1, input_)))

  def __call__(self, inputs: Array) -> Array:
    """Recycle the inputs by running them through the model."""
    return self.recycler.apply(self.params, inputs)

  def _loss_fn(self, param, inputs: Array, targets: Array) -> float:
    """A version of the loss function that passes in the params.

    We have this function because we need the parameters to be an argument
    so we can take the gradient of them.

    Args:
      param: The current parameters.
      inputs: The inputs to the recycler.
      targets: The gold output.

    Returns:
      The loss.
    """
    preds = self.recycler.apply(param, inputs)
    loss_value = optax.l2_loss(preds, targets)
    return loss_value.mean()

  def save(self, path: str):
    """Save the model pytree to `path`."""
    with gfile.GFile(path, "wb") as wf:
      wf.write(flax.serialization.msgpack_serialize(
          flax.core.unfreeze(self.params)))

  def load(self, path: str) -> Recycler:
    """Load the model pytree from `path`."""
    with gfile.GFile(path, "rb") as rf:
      self.params = flax.core.freeze(
          flax.serialization.msgpack_restore(rf.read()))
    return self

  def fit(self,
          inputs: Array,
          targets: Array,
          batch_size: int = 50,
          epochs: int = 25,
          learning_rate: float = 3e-4,
          **kwargs) -> Recycler:
    """Learn weights that map from `inputs` to `targets`."""
    # Use Adam to learn weights.
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(self.params)
    params = self.params

    @jax.jit
    def step(params, opt_state, batch_src, batch_tgt):
      """A single learning step."""
      loss_val, grads = jax.value_and_grad(self._loss_fn)(
          params, batch_src, batch_tgt)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_val

    # Create a rng key for reproducing shuffle order of the dataset.
    rng = jax.random.PRNGKey(0)
    for epoch in range(epochs):
      # Shuffle inputs and targets (in the same way) so each training example is
      # i.i.d.
      # Ratchet the RNG one step.
      shuffle_rng, rng = jax.random.split(rng)
      # Assign a value to each example via a uniform draw.
      # Get the indices that would sort the random values.
      # Use those indices to reorder the full dataset.
      shuffle_indices = jnp.argsort(jax.random.uniform(shuffle_rng,
                                                       (len(inputs),)))
      train_inputs = inputs[shuffle_indices]
      train_targets = targets[shuffle_indices]
      # Extract batches from the shuffled inputs/targets
      for i in range(0, len(inputs), batch_size):
        b_in = train_inputs[i:i+batch_size]
        b_out = train_targets[i:i+batch_size]
        # Update the parameters on this batch.
        params, opt_state, loss_val = step(params, opt_state, b_in, b_out)
        if i % 100 == 0:
          logging.info(
              "Epoch: %d, Step: %d, Loss: %f", epoch, i, loss_val.item())

    # Save the final parameters.
    self.params = params
    return self


class LinearCombinationRecycler(Recycler):
  """Recycle a prompt with a linear combination.

  """

  def __init__(self, source_support: Array, target_support: Array):
    self.source_support = source_support
    self.target_support = target_support

  def __call__(self, inputs: Array) -> Array:
    # self.source_support [T, E]
    # inputs [P, E]
    # self.source_support.T [E, T]
    # inputs.T [E, P]
    # lstsq -> find x st. a @ x = b
    # a = self.source_support.T
    # b = inputs.T
    # self.source_support.T @ x = inputs.T
    # [E, T] @ [T, P] = [E, P]
    # x = [T, P]
    # x.T = [P, T]
    # target_support = [T, E]
    # x.T @ target_support [P, T] @ [T, E] = [P, E]
    x = tf.linalg.lstsq(self.source_support.T, inputs.T)
    return tf.transpose(x) @ self.target_support


def make_recycler(
    method: str, config, source_embeddings: Array, target_embeddings: Array
) -> Recycler:
  """Make the recycler model, dispatched on method."""
  # Get any config arguments for the method
  args = config["recycling_methods"][method].get("__init__", {})
  if method == "jax-nn":
    # Add input and output shape information to the arguments.
    args["input_"] = source_embeddings.shape[-1]
    args["output"] = target_embeddings.shape[-1]
    klass = JaxRecycler
  elif method == "tf-lstsq":
    klass = TFLstSqRecycler
  elif method == "linear-combination":
    klass = LinearCombinationRecycler
    # Add the embeddding tables to the arguments as they are saved and used.
    args["source_support"] = source_embeddings
    args["target_support"] = target_embeddings
  else:
    raise ValueError()
  return klass(**args)


def find_steps(path: str) -> List[int]:
  """Get the sorted list of all checkpoint steps within path."""
  numpy_dir = os.path.join(path, "numpy_checkpoints")
  if not gfile.exists(numpy_dir):
    raise ValueError(f"Could not find any checkpoints within {numpy_dir}.")
  steps = []
  for ckpt in gfile.listdir(numpy_dir):
    if m := re.match(r"checkpoint_(\d+)$", ckpt):
      steps.append(m.groups(1)[0])
  # Sort as integers to get a true sort, would cause issues if we had steps that
  # looked like 0+\d+ as the leading zeros are removed, but we don't.
  return sorted(map(int, steps))


def np_save(path: str, arr):
  """Save a numpy file using gfile for remote locations."""
  with gfile.GFile(path, "wb") as wf:
    np.save(wf, arr)


def main(argv):
  """Run all recycling."""
  del argv

  logging.info("Reading Configuration from: %s", _CONFIG_FILE.value)
  with gfile.GFile(_CONFIG_FILE.value) as f:
    config = json.loads(f.read())

  dataset = config["dataset"]
  skips = set(config.get("combination_skips", []))

  root_dir = config["root_dir"]
  _make_dir(root_dir)
  logging.info("Everything starts from:\n\t%s", root_dir)

  # TODO: Make it possible to control which sources, targets,
  # methods, etc. to use when recycling.
  # Collect and verify that all settings have matching:
  # * methods :: How we are going to recycle the prompts.
  # * seed :: Pretrained model seeds we used.
  # * inits :: How prompt tuning prompts were initialized.
  # * runs :: The multiple runs we have for each trained prompt.
  methods = set(config["recycling_methods"])
  # Enumerate Seeds
  seeds = set(config["pretrained"])
  if set(config["prompts"]) != seeds:
    raise ValueError()
  # Enumerate Inits
  inits = set(config["prompts"][list(seeds)[0]])
  for seed in seeds:
    if set(config["prompts"][seed]) != inits:
      raise ValueError()
  # Enumerate runs
  runs = set(config["prompts"][list(seeds)[0]][list(inits)[0]])
  for seed in seeds:
    for init in inits:
      if set(config["prompts"][seed][init]) != runs:
        raise ValueError()

  embedding_cache = {}

  if _RECYCLE.value:
    # Loop over all the permutations of (to, from) recycling pairs.
    for source_model, target_model in itertools.permutations(seeds, r=2):
      # Skip some combinations listed in the config.
      if (source_model, target_model) in skips:
        continue
      # We iterate over inits before methods so that we don't have to reload the
      # recycler in the method loop. The recycler needs the embeddings and the
      # embeddings need the init
      for init in inits:
        # For the wayward experiments we are only using the embeddings that are
        # present in the embedded discrete prompt string. Add `-wayward` to make
        # it distinct in the cache.
        if init == "wayward":
          method = "from_string"
          source_key = f"{source_model}-wayward"
          target_key = f"{target_model}-wayward"
        else:
          method = "default"
          source_key = source_model
          target_key = target_model
        # Load embeddings that are not in the cache and add them to the cache so
        # we don't keep reloading them.
        if source_key not in embedding_cache:
          logging.info(
              "Loading source embeddings and caching as %s", source_key)
          embedding_cache[source_key] = load_embeddings(
              config["pretrained"][source_model], method, config)
        if target_key not in embedding_cache:
          logging.info(
              "Loading target embeddings and caching as %s", target_key)
          embedding_cache[target_key] = load_embeddings(
              config["pretrained"][target_model], method, config)
        # Get the embeddings as they are now in the cache.
        logging.info(
            "Loading source embeddings from the cache with key: %s", source_key)
        source_embeddings = embedding_cache[source_key]
        logging.info(
            "Source Embedding has a shape of %s", source_embeddings.shape)
        logging.info(
            "Loading target embeddings from the cache with key: %s", target_key)
        target_embeddings = embedding_cache[target_key]
        logging.info(
            "Target Embedding has a shape of %s", target_embeddings.shape)
        # Loop over all the recycler options for this pair.
        for method in methods:
          # Create the recycler, include the source and target embeddings as the
          # linear-combination and jax-nn recycler need them. (The
          # linear-combination needs this as part of the init as it has a no-op
          # save/load).
          #
          # Creating the recycler here lets us save and reuse it across inits,
          # runs, and steps.
          recycler = make_recycler(
              method, config, source_embeddings, target_embeddings)
          # The recycler will get saved to:
          #   ${root}/${method}/learned-recyclers/${recycler_type}
          method_dir = os.path.join(root_dir, method)
          if _make_dir(method_dir):
            logging.info(
                "Everything related to this recycler will live in:\n\t%s",
                method_dir)
          recycler_save_dir = os.path.join(method_dir, "learned-recyclers")
          if _make_dir(recycler_save_dir):
            logging.info("Saving recycler checkpoints in:\n\t%s",
                         recycler_save_dir)
          recycler_type = f"{source_model}_to_{target_model}"
          # Add -wayward to the wayward version as it is different (built on a
          # different source/target embedding table).
          if init == "wayward":
            recycler_type = f"{recycler_type}-wayward"
          recycler_save_path = os.path.join(
              recycler_save_dir,
              recycler_type).replace(" ", "_")
          # If the recycler already exists, load it and run to see the loss.
          if (not config.get("clobber", False) and
              gfile.exists(recycler_save_path)):
            logging.info("Reloading recycler from %s", recycler_save_path)
            recycler = recycler.load(recycler_save_path)
            loss = recycler.loss_fn(source_embeddings, target_embeddings)
            logging.info("Loaded recycler had a loss of %f", loss)
          # Otherwise, fit the recycler using args from the config.
          else:
            logging.info("Fitting recycler.")
            args = config["recycling_methods"][method].get("fit", {})
            recycler = recycler.fit(
                source_embeddings, target_embeddings, **args)
            logging.info("Saving fitted recycler to %s", recycler_save_path)
            recycler.save(recycler_save_path)

          # Loop through each of the prompt training runs.
          for run in runs:
            run_dir = os.path.join(
                method_dir, dataset, init, run).replace(" ", "_")
            if _make_dir(run_dir):
              logging.info("Saving run related values in:\n\t%s", run_dir)
            # Eventually, all the eval runs will be saved in:
            #   ${root}/${method}/${dataset}/${init}/${run}/eval
            eval_dir = os.path.join(run_dir, "eval")
            if _make_dir(eval_dir):
              logging.info("Saving eval runs in:\n\t%s", eval_dir)
            # All the recycled prompts will be saved in:
            #   ${root}/${method}/${dataset}/${init}/${run}/recycled-prompt
            prompt_dir = os.path.join(run_dir, "recycled-prompt")
            if _make_dir(prompt_dir):
              logging.info("Saving recycled prompts to:\n\t%s", prompt_dir)
            steps = config.get("steps", None)
            if steps is None:
              steps = find_steps(config["prompts"][source_model][init][run])
            for step in steps:
              source_prompt_path = os.path.join(
                  config["prompts"][source_model][init][run],
                  "numpy_checkpoints",
                  f"checkpoint_{step}",
                  "encoder.prompt.prompt.prompt")
              logging.info("Loading %s prompt from step %s at %s",
                           source_model,
                           step,
                           source_prompt_path)
              source_prompt = prompts.np_load(source_prompt_path)
              recycled_prompt = recycler(source_prompt)
              recycle_description = (f"recycled_from_{source_model}_at_{step}_"
                                     f"to_{target_model}")
              # A recycled prompt will live at:
              #   ${root}/${method}/${dataset}/${init}/${run}/recycled-prompt/
              #     recycled_from_${source}_at_${step}_to_${target}
              recycled_prompt_path = os.path.join(
                  prompt_dir,
                  recycle_description
              ).replace(" ", "_")
              logging.info(
                  "Saving recycled prompt to %s", recycled_prompt_path)
              np_save(recycled_prompt_path, recycled_prompt)

  # Generation the CLI arguments needed to run all these recycling experiments.
  # We do this in a second loop to make it easier to run without recycling.
  cli_args = []
  for source_model, target_model in itertools.permutations(seeds, r=2):
    # Skip some combinations listed in the config.
    if (source_model, target_model) in skips:
      continue
    for init in inits:
      for method in methods:
        method_dir = os.path.join(root_dir, method)
        for run in runs:
          run_dir = os.path.join(
              method_dir, dataset, init, run).replace(" ", "_")
          steps = config.get("steps", None)
          if steps is None:
            steps = find_steps(config["prompts"][source_model][init][run])
          for step in steps:
            eval_dir = os.path.join(run_dir, "eval")
            prompt_dir = os.path.join(run_dir, "recycled-prompt")
            recycle_description = (f"recycled_from_{source_model}_at_{step}_"
                                   f"to_{target_model}")
            recycled_prompt_path = os.path.join(
                prompt_dir,
                recycle_description).replace(" ", "_")
            # Create the cli arguments that will run recycling for this prompt.
            cli_size = "base"
            if target_model == "Default":
              cli_seed = "default"
            elif target_model in ("Large", "XL", "XXL"):
              cli_seed = "default"
              cli_size = target_model.lower()
            else:
              cli_seed = target_model.split()[-1]
            recycle_eval_path = os.path.join(
                eval_dir, recycle_description).replace(" ", "_")
            command = (f"--seed={cli_seed} --size={cli_size} "
                       f"--prompt_file={recycled_prompt_path} "
                       f"--output_base_dir={recycle_eval_path}")
            cli_args.append(command)
  # Write all flags to disk so we can loop over them.
  cli_file = os.path.join(root_dir, f"cli_args-{dataset}.txt")
  logging.info("Writing CLI flags to trigger %d recycling runs to %s",
               len(cli_args),
               cli_file)
  with gfile.GFile(cli_file, "w") as wf:
    wf.write("\n".join(cli_args))


if __name__ == "__main__":
  app.run(main)
