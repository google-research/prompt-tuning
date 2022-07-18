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

"""Register ReCoRD rank classification tasks."""

import functools

from prompt_tuning.data import features
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.recycling.data import metrics as pr_metrics
import seqio
from t5.data import glue_utils
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow as tf
import tensorflow_datasets as tfds


# Fork of t5.data.preprocessors.record where the entities are passes through and
# we remove the duplication of examples with multiple answers.
def record(dataset):
  """Convert ReCoRD examples to text2text examples.

  ReCoRD contains a passage, query containing a '@placeholder' string, and a set
  of entities that are the possible values of the placeholder. Each train and
  validation example will have a list of answers, any of which would be
  considered correct.

  Note:
    This fork makes sure the entity list is still available in the output
    features and it removes the replication of examples with multiple answers.
    To be used with rank classification.

  Args:
    dataset: a tf.data.Dataset to process.

  Returns:
    a tf.data.Dataset
  """

  def my_fn(x):
    """Converts the processed example to text2text strings."""
    passage = x['passage']
    passage = tf.strings.regex_replace(passage,
                                       r'(\.|\?|\!|\"|\')\n@highlight\n',
                                       r'\1 ')
    passage = tf.strings.regex_replace(passage, r'\n@highlight\n', '. ')

    strs_to_join = [
        'record query:', x['query'], 'entities:',
        tf.strings.reduce_join(x['entities'], separator=', '), 'passage:',
        passage
    ]
    joined = tf.strings.join(strs_to_join, separator=' ')

    ex = {}

    # Store the data index in the returned example (used by eval)
    ex['idx/passage'] = x['idx']['passage']
    ex['idx/query'] = x['idx']['query']

    ex['inputs'] = joined
    # Note that "answers" has been converted to a single string by the
    # process_answers function.
    ex['targets'] = x['answers'][0]
    # Pass-through full list of answers for eval
    ex['answers'] = x['answers']
    # Pass-through the entity list for rank classification
    ex['entities'] = x['entities']
    return ex

  return dataset.map(my_fn, num_parallel_calls=tf.data.AUTOTUNE)


def get_inputs(ex):
  """Get copies of the input for each possible target (pulled from the ex)."""
  # Do the replication via tf ops as the shape of ex["entities"] is dynamic.
  return tf.tile(tf.expand_dims(ex["inputs"], axis=0),
                 (tf.shape(ex["entities"])[0],))


def get_targets(ex):
  """Get the targets, they are the entity list."""
  return ex["entities"]


def get_correct(ex):
  """Boolean mask for each of the possible answers."""
  # Create a |E| x |A| mask. Each row represents an entity and each column an
  # answer. There will be a 1 when an entity and answer match. Then or over the
  # columns (there should only be one 1 at most in a row) to get a multi-hot
  # vector of entities that are in the answers list.
  return tf.math.reduce_any(
      tf.equal(
          tf.expand_dims(ex["entities"], axis=1),
          tf.expand_dims(ex["answers"], axis=0)
      ),
      axis=1
  )

b = tfds.text.super_glue.SuperGlue.builder_configs["record"]

# Training/Validation task
seqio.TaskRegistry.add(
    name="pr:super_glue_record_v102_training",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/record:1.0.2",
        splits={
            "train": "train[:-10000]",
            "validation": "train[-10000:]"
        }
    ),
    preprocessors=[
        # Use the real preprocessor for training so we get the duplicated
        # examples for multiple answers.
        glue_utils.get_glue_text_preprocessor(b),
        pt_preprocessors.remove_first_text_token,
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim
    ],
    postprocess_fn=glue_utils.get_glue_postprocess_fn(b),
    metric_fns=glue_utils.get_super_glue_metric(b.name),
    output_features=features.T5_FEATURES
)

# Generation Eval task
seqio.TaskRegistry.add(
    name="pr:super_glue_record_v102_test",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/record:1.0.2",
        splits=("validation",)
    ),
    preprocessors=[
        glue_utils.get_glue_text_preprocessor(b),
        pt_preprocessors.remove_first_text_token,
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim
    ],
    postprocess_fn=glue_utils.get_glue_postprocess_fn(b),
    # We rely on the new de-duping metrics to handle multiple copies for inputs
    # with multiple answers.
    metric_fns=glue_utils.get_super_glue_metric(b.name),
    output_features=features.T5_FEATURES
)

# Rank classification Eval Task
seqio.TaskRegistry.add(
    name="pr:super_glue_record_v102_rc",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/record:1.0.2",
        splits=("validation",)
    ),
    preprocessors=[
        record,
        pt_preprocessors.remove_first_text_token,
        functools.partial(
            t5_preprocessors.rank_classification,
            inputs_fn=get_inputs,
            targets_fn=get_targets,
            is_correct_fn=get_correct,
            mode="eval"
        ),
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5_postprocessors.rank_classification,
    metric_fns=[
        t5_metrics.rank_classification,
        functools.partial(
            pr_metrics.prefix_metric_names,
            metric_fn=functools.partial(
                t5_metrics.rank_classification,
                normalize_by_target_length=True),
            metric_prefix="normed"
        )],
    output_features=features.T5_FEATURES
)

# Mixture of all tasks
seqio.MixtureRegistry.add(
    "pr:super_glue_record_v102",
    [
        "pr:super_glue_record_v102_training",
        "pr:super_glue_record_v102_test",
        "pr:super_glue_record_v102_rc",
    ],
    default_rate=1.0
)
# Mixture of eval tasks
seqio.MixtureRegistry.add(
    "pr:super_glue_record_v102_eval",
    [
        "pr:super_glue_record_v102_test",
        "pr:super_glue_record_v102_rc",
    ],
    default_rate=1.0
)
