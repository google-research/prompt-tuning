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

"""A feature converter that groups the input with each possible target."""

from typing import Mapping
import seqio
import tensorflow as tf


class ValidLabelOnlyEncDecFeatureConverter(
    seqio.feature_converters.FeatureConverter):
  """EncDec where each is |C| batch, plus a one-hot correct label feature."""

  TASK_FEATURES = {
      "inputs": seqio.feature_converters.FeatureConverter.FeatureSpec(
          dtype=tf.int32,
          rank=2,
          sequence_dim=1),
      "targets": seqio.feature_converters.FeatureConverter.FeatureSpec(
          dtype=tf.int32,
          rank=2,
          sequence_dim=1),
      "is_correct": seqio.feature_converters.FeatureConverter.FeatureSpec(
          dtype=tf.bool),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32,
                                                                rank=2,
                                                                sequence_dim=1),
      "decoder_target_tokens":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32,
                                                                rank=2,
                                                                sequence_dim=1),
      "decoder_input_tokens":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32,
                                                                rank=2,
                                                                sequence_dim=1),
      "decoder_loss_weights":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.int32,
                                                                rank=2,
                                                                sequence_dim=1),
      "is_correct":
          seqio.feature_converters.FeatureConverter.FeatureSpec(dtype=tf.bool)
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "decoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
      "decoder_positions": tf.int32
  }

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    The conversion process involves three steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      # targets_segment_id is present only for a packed dataset.
      decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
          features["targets"],
          sequence_id=features.get("targets_segment_ids", None))

      d = {"encoder_input_tokens": tf.transpose(features["inputs"]),
           "decoder_target_tokens": tf.transpose(features["targets"]),
           "decoder_input_tokens": tf.transpose(decoder_input_tokens),
           # Loss is computed for all but the padding positions.
           "decoder_loss_weights":
               seqio.feature_converters.non_padding_position(
                   tf.transpose(features["targets"])),
           "is_correct": features["is_correct"]}

      if self.pack:
        d["encoder_segment_ids"] = features["inputs_segment_ids"]
        d["decoder_segment_ids"] = features["targets_segment_ids"]
        d["encoder_positions"] = features["inputs_positions"]
        d["decoder_positions"] = features["targets_positions"]

      return d

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]
    correct_length = task_feature_lengths["is_correct"]

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
        "is_correct": correct_length
    }
    if self.pack:
      model_feature_lengths["encoder_segment_ids"] = encoder_length
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["encoder_positions"] = encoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths
