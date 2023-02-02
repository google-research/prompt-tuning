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

"""Rank Classification dataset."""

import copy
import functools
from typing import Sequence
import numpy as np
from prompt_tuning.data import features
from prompt_tuning.data import metrics as pt_metrics
from prompt_tuning.data import postprocessors as pt_postprocessors
from prompt_tuning.data import preprocessors as pt_preprocessors
import seqio
import six
import t5.data
from t5.data import glue_utils
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
import tensorflow as tf
import tensorflow_datasets as tfds


PERCEPTRON_DEFAULT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(),
        add_eos=True,
        rank=2),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(),
        add_eos=True,
        rank=2),
    "is_correct": seqio.Feature(
        vocabulary=seqio.PassThroughVocabulary(size=2),
        add_eos=False,
        dtype=tf.bool)}


@seqio.map_over_dataset
def to_dense(ex, keys: Sequence[str] = ("inputs", "targets")):
  """Convert ragged arrays into dense ones."""
  for key in keys:
    ex[key] = ex[key].to_tensor()
  return ex


@seqio.map_over_dataset
def transpose(ex, keys: Sequence[str] = ("inputs", "targets")):
  """Transpose our arrays for later trimming works."""
  for key in keys:
    ex[key] = tf.transpose(ex[key])
  return ex


def perceptron_postprocessor(decoded_model_output,
                             postprocess_func,
                             *args,
                             **kwargs):
  """Convert batched example to a standard single feature example.

  In order to use our standard postprocessors and metric functions, we need the
  fields of the example of be a single value, instead of the |C| batch we use at
  train time.

  The inputs are replicated for each of the |C| possible targets so we just
  grab the first one.

  The correct target can be extracted from the `is_correct` and
  `targets_pretokenized` features. For processing gold examples, we extract the
  correct target and run the postprocessor on that. For processing model
  prediction, we set the correct value as the `targets_pretokenized` feature
  in case it is used in something like a tensorboard text output.

  Args:
    decoded_model_output: The models prediction. Always a string. Ignored when
      `is_target` kwargs is True.
    postprocess_func: The real postprocessor that we run on the output after
      fixing feature shapes.
    *args: Passed to the real postprocessor.
    **kwargs: Passed to the real postprocessor.

  Returns:
    The output of the postprocess_func called where the batched features in the
    example have been converted to scalar features.
  """
  example = kwargs["example"]
  is_correct = example["is_correct"]
  correct = np.argmax(is_correct).item()
  correct_target = six.ensure_text(example["targets_pretokenized"][correct])
  if kwargs.get("is_target", False):
    return postprocess_func(
        decoded_model_output=correct_target, *args, **kwargs)
  new_example = copy.deepcopy(example)
  new_example["targets_pretokenized"] = correct_target
  new_example["inputs_pretokenized"] = example["inputs_pretokenized"][0]
  kwargs["example"] = new_example
  return postprocess_func(
      decoded_model_output=decoded_model_output, *args, **kwargs)


bs = [
    ("glue", tfds.text.glue.Glue.builder_configs[ds])
    for ds in ("cola", "sst2", "mrpc", "qqp", "qnli", "rte")
] + [
    ("super_glue", tfds.text.super_glue.SuperGlue.builder_configs[ds])
    for ds in ("boolq", "cb", "copa", "rte", "wic")
]

# Slice part of the training data into a validation set to use.
NUM_VAL = {
    # GLUE
    "cola": 500,
    "sst2": 2000,
    "mrpc": 200,
    "qqp": 5000,
    "qnli": 3000,
    "rte": 200,
    # SuperGLUE
    "boolq": 400,
    "cb": 50,
    "copa": 50,
    "wic": 400,
}


def get_version(group_name: str) -> str:
  """Get the version string, 2.0.0 for GLUE tasks and 1.0.2 for SuperGLUE."""
  if group_name.lower() == "glue":
    return "2.0.0"
  elif group_name.lower() == "super_glue":
    return "1.0.2"
  else:
    raise ValueError(f"Unknown group: {group_name}")


def get_seqio_version(tfds_version: str) -> str:
  """Normalize the version to the style used in seqio task names."""
  return f"v{tfds_version.replace('.', '')}"


def get_metric(group_name: str, dataset: str):
  """Proxy to the correct metric getting function based on GLUE vs SuperGLUE."""
  if group_name.lower() == "glue":
    return glue_utils.get_glue_metric(dataset)
  elif group_name.lower() == "super_glue":
    return glue_utils.get_super_glue_metric(dataset)
  else:
    raise ValueError(f"Unknown group: {group_name}")


def get_inputs(ex, labels):
  """Extract |C| inputs from an example."""
  return [ex["inputs"]] * len(labels)


def get_targets(ex, labels):
  """Get the possible labels for an example."""
  # Cast this to a list as some datasets (BoolQ, etc.) are a Tuple which is
  # processed differently when turned into a tensor.
  del ex
  return list(labels)


def get_correct(ex, labels):
  """"Get the one-hot vector marking the correct label."""
  return tf.equal(labels, ex["targets"])


for group, b in bs:
  version = get_version(group)
  seqio_version = get_seqio_version(version)
  # ===== Perceptron Loss Approach =====
  # Note: This data format can be used Cross-Entropy loss model that only
  # considers the valid labels.
  seqio.TaskRegistry.add(
      name=f"{group}_{b.name}_{seqio_version}_perceptron_train",
      source=seqio.TfdsDataSource(
          tfds_name=f"{group}/{b.name}:{version}",
          splits={
              "train": f"train[:-{NUM_VAL[b.name]}]",
              "validation": f"train[-{NUM_VAL[b.name]}:]",
          }),
      preprocessors=[
          glue_utils.get_glue_text_preprocessor(b),
          pt_preprocessors.remove_first_text_token,
          functools.partial(
              preprocessors.rank_classification,
              inputs_fn=functools.partial(get_inputs, labels=b.label_classes),
              targets_fn=functools.partial(get_targets, labels=b.label_classes),
              is_correct_fn=functools.partial(get_correct,
                                              labels=b.label_classes),
              # We want everything as a single batch.
              mode="fewshot_eval"),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
          to_dense,
          # Transpose everything that is rank 2 so the sequence dim is 0 so that
          # the various trimming code built into seqio works. We will transpose
          # back in the feature converter.
          transpose
      ],
      # Our inference will still output only a single label, so use the default
      # postprocessors and metrics (after the postprocessor shim reformats
      # auxiliary features that are used sometimes.).
      postprocess_fn=functools.partial(
          perceptron_postprocessor,
          postprocess_func=functools.partial(
              pt_postprocessors.postprocess_with_examples,
              postprocess_func=glue_utils.get_glue_postprocess_fn(b))),
      metric_fns=[
          functools.partial(pt_metrics.metric_with_examples, func)
          for func in get_metric(group, b.name)
      ] + [functools.partial(pt_metrics.text_examples,
                             task_name=f"{b.name} Perceptron Validation")],
      output_features=PERCEPTRON_DEFAULT_FEATURES
  )

  # Create the validation data as a different task that we will mix together so
  # that we can get both curves on tensorboard even though t5x only evaluates on
  # a single split.
  seqio.TaskRegistry.add(
      name=f"{group}_{b.name}_{seqio_version}_perceptron_test",
      source=seqio.TfdsDataSource(
          tfds_name=f"{group}/{b.name}:{version}",
          splits=("validation",)),
      preprocessors=[
          glue_utils.get_glue_text_preprocessor(b),
          pt_preprocessors.remove_first_text_token,
          functools.partial(
              preprocessors.rank_classification,
              inputs_fn=functools.partial(get_inputs, labels=b.label_classes),
              targets_fn=functools.partial(get_targets, labels=b.label_classes),
              is_correct_fn=functools.partial(get_correct,
                                              labels=b.label_classes),
              # We want everything as a single batch.
              mode="fewshot_eval"),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
          to_dense,
          transpose
      ],
      postprocess_fn=functools.partial(
          perceptron_postprocessor,
          postprocess_func=functools.partial(
              pt_postprocessors.postprocess_with_examples,
              postprocess_func=glue_utils.get_glue_postprocess_fn(b))),
      metric_fns=[
          functools.partial(pt_metrics.metric_with_examples, func)
          for func in get_metric(group, b.name)
      ] + [functools.partial(pt_metrics.text_examples,
                             task_name=f"{b.name} Perceptron Test")],
      output_features=PERCEPTRON_DEFAULT_FEATURES
  )

  # Mix Perceptron training together with Perceptron (rank classification) test.
  seqio.MixtureRegistry.add(
      f"{group}_{b.name}_{seqio_version}_perceptron",
      [f"{group}_{b.name}_{seqio_version}_perceptron_train",
       f"{group}_{b.name}_{seqio_version}_perceptron_test"],
      default_rate=1.0
  )

  # ===== Default Approach to compare against =====
  # Train the Model with x-entropy
  seqio.TaskRegistry.add(
      name=f"{group}_{b.name}_{seqio_version}_generation_train",
      source=seqio.TfdsDataSource(
          tfds_name=f"{group}/{b.name}:{version}",
          splits={
              "train": f"train[:-{NUM_VAL[b.name]}]",
              "validation": f"train[-{NUM_VAL[b.name]}:]",
          }),
      preprocessors=[
          glue_utils.get_glue_text_preprocessor(b),
          pt_preprocessors.remove_first_text_token,
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          pt_postprocessors.postprocess_with_examples,
          glue_utils.get_glue_postprocess_fn(b)),
      metric_fns=[
          functools.partial(pt_metrics.metric_with_examples, func)
          for func in get_metric(group, b.name)
      ] + [functools.partial(pt_metrics.text_examples,
                             task_name=f"{b.name} Generation Valid")],
      output_features=features.T5_FEATURES
  )

  # Create the validation data as a different task that we will mix together so
  # that we can get both curves on tensorboard even though t5x only evaluates on
  # a single split.
  # Evaluate the model with beam decoding.
  seqio.TaskRegistry.add(
      name=f"{group}_{b.name}_{seqio_version}_generation_test",
      source=seqio.TfdsDataSource(
          tfds_name=f"{group}/{b.name}:{version}",
          splits=("validation",)),
      preprocessors=[
          glue_utils.get_glue_text_preprocessor(b),
          pt_preprocessors.remove_first_text_token,
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          pt_postprocessors.postprocess_with_examples,
          glue_utils.get_glue_postprocess_fn(b)),
      metric_fns=[
          functools.partial(pt_metrics.metric_with_examples, func)
          for func in get_metric(group, b.name)
      ] + [functools.partial(pt_metrics.text_examples,
                             task_name=f"{b.name} Generation Test")],
      output_features=features.T5_FEATURES
    )

  # Evaluate the model with rank classification (but trained with x-entropy)
  seqio.TaskRegistry.add(
      name=f"{group}_{b.name}_{seqio_version}_rank_classification_validation",
      source=seqio.TfdsDataSource(
          tfds_name=f"{group}/{b.name}:{version}",
          splits={"validation": f"train[-{NUM_VAL[b.name]}:]"}),
      preprocessors=[
          glue_utils.get_glue_text_preprocessor(b),
          pt_preprocessors.remove_first_text_token,
          functools.partial(
              preprocessors.rank_classification,
              inputs_fn=functools.partial(get_inputs, labels=b.label_classes),
              targets_fn=functools.partial(get_targets, labels=b.label_classes),
              is_correct_fn=functools.partial(get_correct,
                                              labels=b.label_classes),
              mode="eval"),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim
      ],
      postprocess_fn=postprocessors.rank_classification,
      metric_fns=[functools.partial(metrics.rank_classification,
                                    num_classes=len(b.label_classes))],
      output_features=features.T5_FEATURES
  )

  # Evaluate the model with rank classification (but trained with x-entropy)
  seqio.TaskRegistry.add(
      name=f"{group}_{b.name}_{seqio_version}_rank_classification_test",
      source=seqio.TfdsDataSource(
          tfds_name=f"{group}/{b.name}:{version}",
          splits=("validation",)),
      preprocessors=[
          glue_utils.get_glue_text_preprocessor(b),
          pt_preprocessors.remove_first_text_token,
          functools.partial(
              preprocessors.rank_classification,
              inputs_fn=functools.partial(get_inputs, labels=b.label_classes),
              targets_fn=functools.partial(get_targets, labels=b.label_classes),
              is_correct_fn=functools.partial(get_correct,
                                              labels=b.label_classes),
              mode="eval"),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim
      ],
      postprocess_fn=postprocessors.rank_classification,
      metric_fns=[functools.partial(metrics.rank_classification,
                                    num_classes=len(b.label_classes))],
      output_features=features.T5_FEATURES
  )

  # Mix Cross Entropy training together with generation and rank classification
  # validation/test.
  seqio.MixtureRegistry.add(
      f"{group}_{b.name}_{seqio_version}_generation",
      [f"{group}_{b.name}_{seqio_version}_rank_classification_validation",
       f"{group}_{b.name}_{seqio_version}_rank_classification_test",
       f"{group}_{b.name}_{seqio_version}_generation_train",
       f"{group}_{b.name}_{seqio_version}_generation_test"],
      default_rate=1.0
  )
