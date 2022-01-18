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
