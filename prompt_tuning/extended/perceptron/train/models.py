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

"""Models that only consider the scores the model assigns to valid labels.

One model is trained with Cross-Entropy loss where the logits are the scores
assigned to each label. The other model uses a Perceptron Ranking loss.
"""

import functools
import itertools
from typing import Mapping, Optional, MutableMapping, Tuple, Any, Sequence
from absl import logging
import clu.metrics as clu_metrics
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning.extended.perceptron.train import feature_converters
from t5x import losses
from t5x import metrics as metrics_lib
from t5x import models
from flaxformer.types import Array


def fold(batch: Mapping[str, Array]) -> Mapping[str, Array]:
  return jax.tree_map(_fold, batch)


def _fold(arr: Array) -> Array:
  if not isinstance(arr, (jnp.ndarray, np.ndarray)):
    return arr
  b, t, *rest = arr.shape
  return jnp.reshape(arr, (b * t, *rest))


def shape_fold(shape: Sequence[int]) -> Tuple[int, ...]:
  b, t, *rest = shape
  return tuple(itertools.chain((b * t,), rest))


def unfold(batch: Mapping[str, Array], batch_size: int) -> Mapping[str, Array]:
  return jax.tree_map(functools.partial(_unfold, batch_size), batch)


def _unfold(batch_size: int, arr: Array) -> Array:
  if not isinstance(arr, (jnp.ndarray, np.ndarray)):
    return arr
  _, *rest = arr.shape
  return jnp.reshape(arr, (batch_size, -1, *rest))


class ValidLabelsOnlyEncoderDecoderModel(models.EncoderDecoderModel):
  """Models whose logits are the model scores over the possible labels."""

  FEATURE_CONVERTER_CLS = (
      feature_converters.ValidLabelOnlyEncDecFeatureConverter)

  def __init__(self, length_normalize, *args, **kwargs):
    self.length_normalize = length_normalize
    super().__init__(*args, **kwargs)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ):
    input_shapes = dict(input_shapes)  # Make sure values are settable.
    for key in ("encoder_input_tokens",
                "decoder_input_tokens",
                "decoder_target_tokens",
                "decoder_loss_weights"):
      if key in input_shapes:
        input_shapes[key] = shape_fold(input_shapes[key])
    return super().get_initial_variables(
        rng, input_shapes, input_types)

  def _compute_logits(  # pytype: disable=signature-mismatch
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None,
      mutable: bool = False,
      **kwargs,
  ):
    batch_size = batch["encoder_input_tokens"].shape[0]
    # [B, C, T] -> [B * C, T]
    folded_batch = fold(batch)
    logging.info("Batch: %s", jax.tree_map(jnp.shape, batch))
    logging.info("Folded Batch: %s", jax.tree_map(jnp.shape, folded_batch))
    # [B * C, T, V]
    logits = super()._compute_logits(params, folded_batch, dropout_rng, mutable)
    # [B * C, T]
    token_scores = -losses.cross_entropy_with_logits(
        logits,
        common_utils.onehot(
            folded_batch["decoder_target_tokens"],
            logits.shape[-1],
            on_value=1,
            off_value=0),
        z_loss=0.0)[0] * folded_batch["decoder_loss_weights"]
    # [B * C]
    seq_scores = jnp.sum(token_scores, axis=-1)
    if self.length_normalize:
      seq_scores /= jnp.sum(folded_batch["decoder_loss_weights"], axis=-1)
    # [B, C]
    return _unfold(batch_size, seq_scores)

  def predict_batch_with_aux(
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      rng: Optional[jax.random.KeyArray] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    # [B, C]
    logits = self._compute_logits(params,
                                  batch,
                                  dropout_rng=None,
                                  mutable=False)
    # [B]
    best_class = jnp.argmax(logits, axis=-1)
    # [B, tt]
    # We have our best class prediction, now select the associated tgt tensor.
    preds = batch["decoder_target_tokens"][
        jnp.arange(best_class.shape[0]), best_class]
    return preds, {}


class CrossEntropyEncoderDecoderModel(ValidLabelsOnlyEncoderDecoderModel):
  """Train with Cross Entropy, but only consider valid labels."""

  def loss_fn(
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None
  ) -> Tuple[jnp.ndarray, models.MetricsMap]:
    # [B, C]
    logits = self._compute_logits(params, batch, dropout_rng, False)

    # Convert the one-hot label representation to class indices, this will be
    # converted back to one-hot which is a waste but this is the format the
    # loss function wants.
    labels = jnp.argmax(batch["is_correct"], axis=-1)  # [B]
    # Each position is valid.
    weights = jnp.ones_like(labels)

    loss, z_loss, _ = losses.compute_weighted_cross_entropy(
        logits,  # [B, C]
        targets=labels,  # [B]
        weights=weights,
        label_smoothing=self._label_smoothing,
        z_loss=self._z_loss,
        loss_normalizing_factor=None)
    metrics = self._compute_metrics(
        logits=logits,
        targets=labels,
        mask=weights,
        loss=loss,
        z_loss=z_loss)
    return loss, metrics


class PerceptronLossEncoderDecoderModel(ValidLabelsOnlyEncoderDecoderModel):
  """Train with a Perceptron ranking loss on valid label outputs."""

  def __init__(self, hinge, *args, **kwargs):
    self.hinge = hinge
    super().__init__(*args, **kwargs)

  def loss_fn(
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, models.MetricsMap]:
    # [B, C]
    logits = self._compute_logits(params, batch, dropout_rng, False)
    # [B] best scores.
    y_hat = jnp.max(logits, axis=-1)
    # [B, C] * [B, C] mask sum (gold score only one left, rest are zeros)
    # -> [B] scores
    y = jnp.sum(logits * batch["is_correct"], axis=-1)
    # [B]
    # Perceptron and hinge loss.
    loss = jnp.maximum(0, self.hinge + (y_hat - y))
    loss = jnp.sum(loss)
    metrics = {
        "loss": metrics_lib.AveragePerStep(total=loss),
        "accuracy": clu_metrics.Accuracy.from_model_output(
            logits=logits, labels=jnp.argmax(batch["is_correct"], axis=-1))
    }
    return loss, metrics
