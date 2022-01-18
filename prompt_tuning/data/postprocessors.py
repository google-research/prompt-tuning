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

"""Custom Prompt Tuning postprocessors.

There are 3 main postprocessors. The first is the `postprocesses_with_examples`
which wraps any other postprocessor function and save the results (as well as
other provided keys) into a dict instead of just returning the postprocessed
result.

There is also a postprocessor that removes the <extra_id_0> from the beginning
of an output. (This is used for the span checkpoints).

There are also postprocessors that translate between the output labels for the
qqp and mrpc tasks, used in testing the transferability of models.
"""

from typing import Sequence, Mapping, Any, Union
from prompt_tuning.data import constants


def remove_prefix(s: str, prefix: str) -> str:
  """Remove prefix from the beginning of the string if present."""
  if s.startswith(prefix):
    return s[len(prefix):]
  return s[:]


# This function is a t5 postprocessor function so it needs to take *args and
# **kwargs, even though it doesn't use them itself.
def remove_extra_id_0(decoded_model_output: str, *args, **kwargs) -> str:  # pylint: disable=unused-argument
  """Possibly remove `<extra_id_0>` and leading white space."""
  return remove_prefix(decoded_model_output, constants.EXTRA_ID_0).lstrip()


def postprocess_with_examples(
    postprocess_func,
    decoded_model_output,
    prediction_field: str = constants.PREDICTION,
    prediction_text_field: str = constants.PREDICTION_TEXT,
    example_fields: Sequence[str] = (constants.INPUT_TEXT,
                                     constants.TARGET_TEXT),
    **kwargs,
) -> Union[Mapping[str, Any], Any]:
  """Wrap a postprocessing function so that it returns a dictionary with extra information.

  Args:
    postprocess_func: The actual postprocessing function you are going to use.
    decoded_model_output: The output from the model.
    prediction_field: The dictionary key to save the postprocessed output to.
    prediction_text_field: The dictionary key to save the model output to.
    example_fields: Fields from the example to copy into the output.
    **kwargs: Extra arguments. One should be `example` which has the unprocessed
      batch that has things like the raw text in it.

  Returns:
    A mapping that includes the postprocessed prediction, the raw prediction
    text, and any extra fields from the example you want.
  """
  if kwargs.get("is_target", False):
    # Our targets are going to stay their normal type so follow the old behavior
    # of just calling the postprocess function on them.
    return postprocess_func(decoded_model_output, **kwargs)
  result = {
      prediction_field: postprocess_func(decoded_model_output, **kwargs),
      prediction_text_field: decoded_model_output
  }
  for field in example_fields:
    result[field] = kwargs.get("example", {}).get(field)
  return result


def sequential(*funcs):
  """Execute a sequence of functions as if it was one function."""

  def execute(x, *args, **kwargs):
    for fun in funcs:
      x = fun(x, *args, **kwargs)
    return x

  return execute


# ========== Zero-Shot Postprocessing ==========
# These are postprocessing functions for t5 which are passed both *args and
def mrpc_to_qqp(decoded_model_output: str, *args, **kwargs) -> str:  # pylint: disable=unused-argument
  """Rewrite mrpc model predictions to the qqp target vocab."""
  return decoded_model_output.replace(constants.EQUIVALENT, constants.DUPLICATE)


def qqp_to_mrpc(decoded_model_output: str, *args, **kwargs) -> str:  # pylint: disable=unused-argument
  """Rewrite qqp model predictions to the mrpc target vocab."""
  return decoded_model_output.replace(constants.DUPLICATE, constants.EQUIVALENT)
