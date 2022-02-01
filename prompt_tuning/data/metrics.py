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

"""T5 metric functions that output text examples.

These metrics functions are designed to understand that the output from the
postprocssing function will be a dict with the prediction as one field and text
examples for the others.

There is a metric function wrapper that extracts predictions from a dict and
hands it off to a real metric function. There are also metric functions that
produce a Text object that will be written to tensorboard.
"""

import itertools
import random
import textwrap
from typing import Sequence, Optional, Dict, List, Mapping, Any
from prompt_tuning.data import constants
import seqio
import six


def metric_with_examples(metric_fn,
                         targets,
                         predictions,
                         prediction_field: str = constants.PREDICTION):
  """Run a metric function where the predictions include extra information."""
  return metric_fn(targets, [p[prediction_field] for p in predictions])


# pylint: disable=unused-argument
def format_classification(indices: Sequence[int],
                          examples: List[Mapping[str, Any]],
                          input_field: str = constants.INPUT_TEXT,
                          target_field: str = constants.TARGET_TEXT,
                          prediction_field: str = constants.PREDICTION_TEXT,
                          **kwargs) -> str:
  """Format text output for classification tasks."""
  texts = []
  for idx in indices:
    example_text = textwrap.dedent(f"""
         example: {six.ensure_text(examples[idx][input_field])}\n
         target: {six.ensure_text(examples[idx][target_field])}\n
         prediction: {six.ensure_text(examples[idx][prediction_field])}\n
         """.lstrip("\n"))
    texts.append(example_text)
  return "\n\n".join(texts)


# pylint: disable=unused-argument
def format_qa(indices: Sequence[int],
              examples: List[Mapping[str, Any]],
              context_field: str = constants.CONTEXT_TEXT,
              question_field: str = constants.QUESTION_TEXT,
              answers_field: str = constants.ANSWERS_TEXT,
              prediction_field: str = constants.PREDICTION_TEXT,
              **kwargs) -> str:
  """Format text output for qa tasks."""
  texts = []
  for idx in indices:
    example_text = textwrap.dedent(f"""
      context: {six.ensure_text(examples[idx][context_field])}\n
      question: {six.ensure_text(examples[idx][question_field])}\n
      answers: {"; ".join([six.ensure_text(a) for a in examples[idx][answers_field]])}\n
      prediction: {six.ensure_text(examples[idx][prediction_field])}\n
      """.lstrip("\n"))
    texts.append(example_text)
  return "\n\n".join(texts)


def label_set_stats(
    targets,
    predictions,
    task_name: str,
    target_field: str = constants.TARGET_TEXT,
    prediction_field: str = constants.PREDICTION_TEXT,
    display_sets: bool = False,
) -> Dict[str, seqio.metrics.Text]:
  """Measure the number of invalid predictions.

  The "target label set" should not change through evaluation, it is just the
  set of gold labels in your validation split.

  The "prediction label set" will change as your model predicts different values
  between epochs.

  "target ∩ prediction" != "len(target)" means your model is systematically
  missing some label. "target - prediction" != 0 means the same thing.

  "prediction - target" != 0 means that your model is hallucinating a label that
  is not included in the target label set.

  Args:
    targets: The postprocessed targets, unused.
    predictions: The postprocessed predictions, as dicts.
    task_name: The name of the task these predictions came from.
    target_field: The dict key to access the pre-tokenized target.
    prediction_field: The dict key to access the pre-tokenized prediction.
    display_sets: Whether to display the actual values of the set of
      target/prediction labels.

  Returns:
    A Text metric with summary information about the sets of gold and predicted
    labels.
  """
  del targets
  target_labels = {six.ensure_text(p[target_field]) for p in predictions}
  prediction_labels = {six.ensure_text(p[prediction_field])
                       for p in predictions}
  outputs = [
      f"target label set cardinality: {len(target_labels)}",
      f"prediction label set cardinality: {len(prediction_labels)}",
      f"target ∩ prediction cardinality: {len(target_labels & prediction_labels)}",
      f"target - prediction cardinality: {len(target_labels - prediction_labels)}",
      f"prediction - target cardinality: {len(prediction_labels - target_labels)}",
  ]
  if display_sets:
    sets = [
        f"target label set: {target_labels}",
        f"prediction label set: {prediction_labels}",
        f"target ∩ prediction: {target_labels & prediction_labels}",
        f"target - prediction: {target_labels - prediction_labels}",
        f"prediction - target: {prediction_labels - target_labels}",
    ]
    # Interleave the cardinality and set values in the output list.
    outputs = list(itertools.chain(*zip(outputs, sets)))
  return {
      f"{task_name} label stats": seqio.metrics.Text(
          textdata="\n\n".join(outputs)
      )
  }


def safe_sample(num_examples: int,
                population: List[Mapping[str, Any]],
                seed: Optional[int] = None) -> Sequence[int]:
  if seed:
    random.seed(seed)
  if num_examples == -1 or num_examples > len(population):
    return range(len(population))
  return random.sample(range(len(population)), k=num_examples)


def text_examples(
    targets,  # pylint: disable=unused-argument
    predictions,
    task_name: str,
    num_examples: int = 8,
    seed: Optional[int] = None,
    find_negative: bool = True,
    format_fn=format_classification,
    input_field: str = constants.INPUT_TEXT,
    target_field: str = constants.TARGET_TEXT,
    prediction_field: str = constants.PREDICTION,
    prediction_text_field: str = constants.PREDICTION_TEXT,
    context_field: str = constants.CONTEXT_TEXT,
    question_field: str = constants.QUESTION_TEXT,
    answers_field: str = constants.ANSWERS_TEXT,
) -> Dict[str, seqio.metrics.MetricValue]:
  """Build a string with the inputs, outputs, and targets for tensorboard."""
  random.seed(seed)

  # We always output a random sample of predictions.
  indices = safe_sample(num_examples, predictions)
  result = {
      f"{task_name} samples":
          seqio.metrics.Text(
              textdata=format_fn(
                  indices,
                  predictions,
                  input_field=input_field,
                  target_field=target_field,
                  prediction_field=prediction_text_field,
                  context_field=context_field,
                  question_field=question_field,
                  answers_field=answers_field))
  }

  # Do we want to mine for negative examples to help see what is going on?
  if find_negative:
    # These are an approximation of the examples we got wrong, some datasets
    # like SQuAD where the targets is a list will always have a mismatch.
    wrongs = [
        p for p, t in zip(predictions, targets) if p[prediction_field] != t
    ]
    indices = safe_sample(num_examples, wrongs)

    result[f"{task_name} negative samples"] = seqio.metrics.Text(
        format_fn(
            indices,
            wrongs,
            input_field=input_field,
            target_field=target_field,
            prediction_field=prediction_text_field,
            context_field=context_field,
            question_field=question_field,
            answers_field=answers_field))
  return result
