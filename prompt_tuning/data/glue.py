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

"""Special versions of Glue Tasks.

The most important formats are:

 * glue_{name}_v002_examples
 * taskless_glue_{name}_v002_examples
 * task_index_glue_{name}_v002_examples

Any task that ends with `_examples` outputs some examples to tensorboard. Any
task that starts with `taskless` doesn't have the initial dataset text token.
Any task that starts with `task_index` has a single task index token prepended
to the tokenized inputs.

Tasks can also have `mt5_` or `byt5_` prepended meaning they use the vocabulary
associated with those models. No prefix means it uses the t5 vocabulary.
"""

import functools

from prompt_tuning.data import features
from prompt_tuning.data import metrics as pt_metrics
from prompt_tuning.data import postprocessors as pt_postprocessors
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.data import utils
import seqio
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_glue_weight_mapping
import tensorflow_datasets as tfds

glue_indexer = utils.task_mapping(
    tuple(b.name for b in tfds.text.glue.Glue.builder_configs.values()),
    {
        "ax": "mnli",
        "mnli_mismatched": "mnli",
        "mnli_matched": "mnli"
    },
)

# Create newer versions of the tasks so that newer tfds versions can build.
# Include older versions to match t5.
for data_ver, task_ver in (("1.0.0", "v002"), ("2.0.0", "v200")):
  for model_prefix, feats in features.MODEL_TO_FEATURES.items():
    # ========== GLUE ==========
    for log_examples in (True, False):
      for b in tfds.text.glue.Glue.builder_configs.values():
        if log_examples:
          postprocess_fn = functools.partial(
              pt_postprocessors.postprocess_with_examples,
              get_glue_postprocess_fn(b))
          metric_fns = [
              functools.partial(pt_metrics.metric_with_examples, func)
              for func in get_glue_metric(b.name)
          ] + [functools.partial(pt_metrics.text_examples, task_name=b.name)]
          examples_suffix = "_examples"
        else:
          postprocess_fn = get_glue_postprocess_fn(b)
          metric_fns = get_glue_metric(b.name)
          examples_suffix = ""
        if log_examples:
          # The non-log_examples version is already defined in other packages.
          # This version of GLUE can output examples to TensorBoard.
          seqio.TaskRegistry.add(
              f"{model_prefix}glue_{b.name}_{task_ver}{examples_suffix}",
              source=seqio.TfdsDataSource(
                  tfds_name=f"glue/{b.name}:{data_ver}",
                  splits=["test"] if b.name == "ax" else None),
              preprocessors=[
                  get_glue_text_preprocessor(b),
                  seqio.preprocessors.tokenize,
                  seqio.CacheDatasetPlaceholder(),
                  seqio.preprocessors.append_eos_after_trim,
              ],
              postprocess_fn=postprocess_fn,
              metric_fns=metric_fns,
              output_features=feats)

        # This version doesn't have the dataset initial text token.
        seqio.TaskRegistry.add(
            f"{model_prefix}taskless_glue_{b.name}_{task_ver}{examples_suffix}",
            source=seqio.TfdsDataSource(
                tfds_name=f"glue/{b.name}:{data_ver}",
                splits=["test"] if b.name == "ax" else None),
            preprocessors=[
                get_glue_text_preprocessor(b),
                pt_preprocessors.remove_first_text_token,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            postprocess_fn=postprocess_fn,
            metric_fns=metric_fns,
            output_features=feats)

        # This version has a task index as the first integer token
        seqio.TaskRegistry.add(
            f"{model_prefix}task_index_glue_{b.name}_{task_ver}{examples_suffix}",
            source=seqio.TfdsDataSource(
                tfds_name=f"glue/{b.name}:{data_ver}",
                splits=["test"] if b.name == "ax" else None),
            preprocessors=[
                get_glue_text_preprocessor(b),
                pt_preprocessors.remove_first_text_token,
                seqio.preprocessors.tokenize,
                functools.partial(
                    pt_preprocessors.add_sentinel_to_beginning,
                    field="inputs",
                    offset=glue_indexer[b.name]),
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            postprocess_fn=postprocess_fn,
            metric_fns=metric_fns,
            output_features=feats)

      # ========== MNLI Mixtures ==========
      # This is a mixture of mnli and the matched/mismatched dev/testing sets.
      # The tfds data for MNLI has the following splits, `train`,
      # `validation_matched`, `validation_mismatched`, `test_matched`,
      # `test_mismatached`. These don't follow the normal naming for splits for
      # other tasks and would cause problems at evaluation time. The tfds data
      # for `MNLIm` however just has `validation` and `test` (which match the
      # `_matched` versions in the MNLI dataset). So when we create this mixture
      # it results in a task that has a `train` split (the mnli training
      # dataset) and two tasks that have `validation` (mnli_matched and
      # mnli_mismatched). The seqio.evaluator is setup so that both of these
      # validation tasks will be used.

      # Replace v002 with `task_ver`. If task_ver is v200 it will get replaced
      # in the task. If the task ver is v002 it will result in the original
      # task name.

      if log_examples:
        # This mixture is MNLI that outputs to TensorBoard.
        seqio.MixtureRegistry.add(
            f"{model_prefix}glue_mnli_and_dev_{task_ver}{examples_suffix}", [
                f"{model_prefix}{x.replace('v002', task_ver)}{examples_suffix}"
                for x in get_glue_weight_mapping().keys()
                if "mnli" in x
            ],
            default_rate=1.0)
      else:
        # There is no non-log examples version of mnli v200
        continue

      # This mixture is MNLI that doesn't have the initial dataset text token.
      seqio.MixtureRegistry.add(
          f"{model_prefix}taskless_glue_mnli_and_dev_{task_ver}{examples_suffix}",
          [
              f"{model_prefix}{x.replace('v002', task_ver)}{examples_suffix}"
              for x in get_glue_weight_mapping().keys()
              if "mnli" in x
          ],
          default_rate=1.0)

      # This mixture is MNLI that has a task_index for the first integer token.
      seqio.MixtureRegistry.add(
          f"{model_prefix}task_index_glue_mnli_and_dev_{task_ver}{examples_suffix}",
          [
              f"{model_prefix}{x.replace('v002', task_ver)}{examples_suffix}"
              for x in get_glue_weight_mapping().keys()
              if "mnli" in x
          ],
          default_rate=1.0)
