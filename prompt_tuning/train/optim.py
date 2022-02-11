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

"""A fork of flax.optim.MultiOptimizer that works with t5x.adafactor.

The original flax.optim.MultiOptimizer can be found at
https://github.com/google/flax/blob/main/flax/optim/base.py
"""

from typing import Sequence, Tuple
import flax
from flax import optim
from flax import traverse_util
from flax.core.frozen_dict import freeze
from flax.core.frozen_dict import unfreeze
import jax
import jax.numpy as jnp
from t5x import adafactor

# pylint: disable=protected-access

# local imports from optim:
OptimizerDef = optim.OptimizerDef
OptimizerState = optim.OptimizerState
Optimizer = optim.Optimizer


class _Marker:
  """Used to mark unoptimized leaves."""

  def __init__(self):
    self._indices = []


def standard_logical_factor_rules(rules=None):
  """Add prompt adafactor rules to your set of rules."""
  if rules is None:
    rules = adafactor.standard_logical_factor_rules()
  rules = unfreeze(rules)
  rules['prompt'] = adafactor.FactorDim.NONE
  rules['tasks'] = adafactor.FactorDim.NONE
  rules['prompt+embed'] = adafactor.FactorDim.NONE
  return freeze(rules)


def tree_of_paths(tree):
  """Converts a (frozen) nested dictionary into a (frozen) dict of paths."""
  is_frozen = isinstance(tree, flax.core.frozen_dict.FrozenDict)
  flat_tree = traverse_util.flatten_dict(unfreeze(tree))
  path_tree = traverse_util.unflatten_dict(
      {k: '/'.join(k) for k in flat_tree.keys()})
  if is_frozen:
    path_tree = freeze(path_tree)
  return path_tree


def subtree_from_traversal(traversal, tree):
  """Creates a (frozen) tree subset given a traversal."""
  is_frozen = isinstance(tree, flax.core.frozen_dict.FrozenDict)
  flat_tree = {}
  for path, leaf in zip(traversal.iterate(tree_of_paths(tree)),
                        traversal.iterate(tree)):
    flat_tree[path] = leaf
  new_tree = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_tree.items()})
  if is_frozen:
    new_tree = freeze(new_tree)
  return new_tree


def update_subtree_of_traversal(traversal, tree, update):
  """Updates a (frozen) tree's subset given a traversal and update subtree."""
  is_frozen = isinstance(tree, flax.core.frozen_dict.FrozenDict)
  flat_tree = traverse_util.flatten_dict(unfreeze(tree))
  flat_tree = {'/'.join(k): v for k, v in flat_tree.items()}
  for path, leaf in zip(traversal.iterate(tree_of_paths(update)),
                        traversal.iterate(update)):
    flat_tree[path] = leaf
  nested_d = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_tree.items()})
  if is_frozen:
    nested_d = freeze(nested_d)
  return nested_d


class MultiOptimizer(OptimizerDef):
  """Optimize subsets of parameters.

  A MultiOptimizer is subclass of :class:`OptimizerDef` and useful for applying
  separate optimizer algorithms to various subsets of the model parameters.

  The example below creates two optimizers using
  :class:`flax.traverse_util.ModelParamTraversal`:
  one to optimize ``kernel`` parameters and to optimize ``bias`` parameters.
  Note each optimizer is created with a different learning rate::

   kernels = traverse_util.ModelParamTraversal(lambda path, _: 'kernel' in path)
   biases = traverse_util.ModelParamTraversal(lambda path, _: 'bias' in path)
   kernel_opt = optim.Momentum(learning_rate=0.01)
   bias_opt = optim.Momentum(learning_rate=0.1)
   opt_def = MultiOptimizer((kernels, kernel_opt), (biases, bias_opt))
   optimizer = opt_def.create(model)

  In order to train only a subset of the parameters, you can simply use a single
  :class:`flax.traverse_util.ModelParamTraversal` instance.

  If you want to update the learning rates of both optimizers online with
  different learning rate schedules, you should update the learning rates when
  applying the gradient. In the following example, the second optimizer is not
  doing any optimization during the first 1000 steps::

    hparams = optimizer.optimizer_def.hyper_params
    new_optimizer = optimizer.apply_gradient(
        grads,
        hyper_params=[
          hparams[0].replace(learning_rate=0.2),
          hparams[1].replace(learning_rate=jnp.where(step < 1000, 0., lr)),
        ])
  """

  def __init__(
      self,
      traversals_and_optimizers: Sequence[
          Tuple[traverse_util.Traversal, OptimizerDef]]):
    """Create a new MultiOptimizer.

    See docstring of :class:`MultiOptimizer` for more details.

    Args:
      traversals_and_optimizers: pairs of flax.traverse_util.Traversal and
      `flax.optim.OptimizerDef` instances.
    """
    traversals, sub_optimizers = zip(*traversals_and_optimizers)
    hyper_params = [opt.hyper_params for opt in sub_optimizers]
    super().__init__(hyper_params)
    self.traversals = traversals
    self.sub_optimizers = sub_optimizers

  def init_state(self, params):
    param_states = jax.tree_map(lambda x: _Marker(), params)
    overlap = False
    for idx, traversal in enumerate(self.traversals):
      for match in traversal.iterate(param_states):
        match._indices.append(idx)
        overlap |= len(match._indices) > 1
    if overlap:
      raise ValueError(
          'Multiple optimizers match the same leaves : ' +
          str(jax.tree_map(lambda match: match._indices, param_states)))

    param_states = jax.tree_map(lambda x: _Marker(), params)
    for focus, opt_def in zip(self.traversals, self.sub_optimizers):
      ps = subtree_from_traversal(focus, params)
      ss = opt_def.init_state(ps)
      param_states = update_subtree_of_traversal(
          focus, param_states, ss.param_states)
    # Update state to None when param is not optimized by any sub optimizer.
    param_states = jax.tree_map(
        lambda x: None if isinstance(x, _Marker) else x, param_states)
    return OptimizerState(jnp.asarray(0, dtype=jnp.int32), param_states)

  def apply_gradient(self, hyper_params, params, state, grads):
    new_params = params
    it = zip(self.traversals, self.sub_optimizers, hyper_params)
    new_param_states = jax.tree_map(lambda x: _Marker(), params)
    for focus, opt_def, hp in it:
      ps = subtree_from_traversal(focus, params)
      gs = subtree_from_traversal(focus, grads)
      ss = subtree_from_traversal(focus, state.param_states)
      prev_ss = OptimizerState(state.step, ss)
      new_ps, new_ss = opt_def.apply_gradient(hp, ps, prev_ss, gs)
      new_params = update_subtree_of_traversal(focus, new_params, new_ps)
      new_param_states = update_subtree_of_traversal(
          focus, new_param_states, new_ss.param_states)
    # Update state to None when param is not optimized by any sub optimizer.
    new_param_states = jax.tree_map(
        lambda x: None if isinstance(x, _Marker) else x, new_param_states)
    return new_params, OptimizerState(state.step + 1, new_param_states)

  def update_hyper_params(self, **hyper_param_overrides):
    """Updates the hyper parameters with a set of overrides.

    This method is called from :meth:`Optimizer.apply_gradient` to create the
    hyper parameters for a specific optimization step.
    MultiOptimizer will apply the overrides for each sub optimizer.

    Args:
      **hyper_param_overrides: the hyper parameters updates
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      The new hyper parameters.
    """
    hps = hyper_param_overrides.pop('hyper_params', self.hyper_params)
    if hyper_param_overrides:
      hps = [hp.replace(**hyper_param_overrides) for hp in hps]
    return hps

  def set_param_axes(self, param_logical_axes):
    """Derives factorization rules from model parameter logical axes."""
    for focus, opt_def in zip(self.traversals, self.sub_optimizers):
      pla_subtree = subtree_from_traversal(focus, param_logical_axes)
      if hasattr(opt_def, 'set_param_axes'):
        opt_def.set_param_axes(pla_subtree)

  def derive_logical_axes(self, optimizer, param_logical_axes):
    """Derives optimizer logical partitioning from model logical partitions."""
    param_states = jax.tree_map(
        lambda x: _Marker(), optimizer.state.param_states)
    for focus, opt_def in zip(self.traversals, self.sub_optimizers):
      if hasattr(opt_def, 'derive_logical_axes'):
        ps = subtree_from_traversal(focus, param_logical_axes)
        ss = subtree_from_traversal(focus, optimizer.state.param_states)
        new_opt = opt_def.derive_logical_axes(
            Optimizer(opt_def, OptimizerState(None, ss), ps), ps)
        param_states = update_subtree_of_traversal(
            focus, param_states, new_opt.state.param_states)
    # Update axes to None when param is not optimized by any sub optimizer.
    param_states = jax.tree_map(
        lambda x: None if isinstance(x, _Marker) else x, param_states)
    return Optimizer(optimizer.optimizer_def,
                     OptimizerState(None, param_states),
                     param_logical_axes)
