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

"""Tasks and Mixture for zeroshot transfer between the QQP and MRPC dataset.

The Quora Question Pairs (QQP) dataset and the Microsoft Research Paraphrase
Corpus are two of the most similar datasets in Glue. They are both have two
sentences as input and the goal of the task is to decide if the two sentences
are duplicated of each other are not.

The frustrating thing about these datasets is they do not have a matching
vocabulary. The QQP dataset uses the strings `"duplicate"` and
`"not_duplicate"`, while the MRPC dataset uses the strings `"equivalent"` and
`"not_equivalent"`. Luckily for us, we can add a simple postprocessing function
 to our task output to convert between the output spaces before actually running
metrics.

Tasks for zeroshot evaluation of task transfer, as well as a mixture that allows
for training on one dataset and zeroshot evaluation on the other.

We don't include a setting where you have the training and validation data for
one dataset but the postprocessing function that expects outputs in the other
dataset's output vocabulary. This is because the targets will still be in the in
original datasets output label space and will cause a mismatch between training
and testing.
"""
import functools

from prompt_tuning.data import features
# Make sure we have access to glue tasks for the training data.
from prompt_tuning.data import glue  # pylint: disable=unused-import
from prompt_tuning.data import metrics as pt_metrics
from prompt_tuning.data import postprocessors as pt_postprocessors
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.data import utils
import seqio
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
import tensorflow_datasets as tfds

glue_indexer = utils.task_mapping(
    tuple(b.name for b in tfds.text.glue.Glue.builder_configs.values()),
    {
        "ax": "mnli",
        "mnli_mismatched": "mnli",
        "mnli_matched": "mnli"
    },
)

# A mapping from (src, tgt) pairs to postprocessing label translation functions.
TRANSLATIONS = {
    ("qqp", "mrpc"): pt_postprocessors.qqp_to_mrpc,
    ("mrpc", "qqp"): pt_postprocessors.mrpc_to_qqp,
}

for model_prefix, feats in features.MODEL_TO_FEATURES.items():
  for source, target in (("qqp", "mrpc"), ("mrpc", "qqp")):
    src = tfds.text.glue.Glue.builder_configs[source]
    tgt = tfds.text.glue.Glue.builder_configs[target]
    translation = TRANSLATIONS[(source, target)]
    for log_examples in (True, False):
      if log_examples:
        postprocess_fn = pt_postprocessors.sequential(
            translation,
            functools.partial(pt_postprocessors.postprocess_with_examples,
                              get_glue_postprocess_fn(tgt)))
        metric_fns = [
            functools.partial(pt_metrics.metric_with_examples, func)
            for func in get_glue_metric(tgt.name)
        ] + [functools.partial(pt_metrics.text_examples, task_name=tgt.name)]
        examples_suffix = "_examples"
      else:
        qqp_postprocess_fn = pt_postprocessors.sequential(
            translation, get_glue_postprocess_fn(tgt))
        metric_fns = get_glue_metric(tgt.name)
        examples_suffix = ""
      # These tasks are the validation datasets, but have a post process
      # function that expects the model output to be a different label space.
      seqio.TaskRegistry.add(
          f"{model_prefix}glue_{target}_from_{source}_v002{examples_suffix}",
          source=seqio.TfdsDataSource(
              tfds_name=f"glue/{tgt.name}:1.0.0", splits=["validation"]),
          preprocessors=[
              get_glue_text_preprocessor(tgt),
              seqio.preprocessors.tokenize,
              seqio.CacheDatasetPlaceholder(),
              seqio.preprocessors.append_eos_after_trim,
          ],
          postprocess_fn=postprocess_fn,
          metric_fns=metric_fns,
          output_features=feats)

      # This version omits the initial dataset text token.
      seqio.TaskRegistry.add(
          f"{model_prefix}taskless_glue_{target}_from_{source}_v002{examples_suffix}",
          source=seqio.TfdsDataSource(
              tfds_name=f"glue/{tgt.name}:1.0.0", splits=["validation"]),
          preprocessors=[
              get_glue_text_preprocessor(tgt),
              pt_preprocessors.remove_first_text_token,
              seqio.preprocessors.tokenize,
              seqio.CacheDatasetPlaceholder(),
              seqio.preprocessors.append_eos_after_trim,
          ],
          postprocess_fn=postprocess_fn,
          metric_fns=metric_fns,
          output_features=feats)

      # This version includes a task index as the first integer token.
      seqio.TaskRegistry.add(
          f"{model_prefix}task_index_glue_{target}_from_{source}_v002{examples_suffix}",
          source=seqio.TfdsDataSource(
              tfds_name=f"glue/{tgt.name}:1.0.0", splits=["validation"]),
          preprocessors=[
              get_glue_text_preprocessor(tgt),
              pt_preprocessors.remove_first_text_token,
              seqio.preprocessors.tokenize,
              functools.partial(
                  pt_preprocessors.add_sentinel_to_beginning,
                  field="inputs",
                  offset=glue_indexer[tgt.name]),
              seqio.CacheDatasetPlaceholder(),
              seqio.preprocessors.append_eos_after_trim,
          ],
          postprocess_fn=postprocess_fn,
          metric_fns=metric_fns,
          output_features=feats)

      # ========== Zero-Shot Mixture ==========
      seqio.MixtureRegistry.add(
          f"{model_prefix}glue_{target}_from_{source}_zeroshot_v002{examples_suffix}",
          [
              f"{model_prefix}glue_{source}_v002",
              f"{model_prefix}glue_{target}_from_{source}_v002{examples_suffix}"
          ],
          default_rate=1.0)
      seqio.MixtureRegistry.add(
          f"{model_prefix}taskless_glue_{target}_from_{source}_zeroshot_v002{examples_suffix}",
          [
              f"{model_prefix}taskless_glue_{source}_v002",
              f"{model_prefix}taskless_glue_{target}_from_{source}_v002{examples_suffix}"
          ],
          default_rate=1.0)

      seqio.MixtureRegistry.add(
          f"{model_prefix}task_index_glue_{target}_from_{source}_zeroshot_v002{examples_suffix}",
          [
              f"{model_prefix}task_index_glue_{source}_v002",
              f"{model_prefix}task_index_glue_{target}_from_{source}_v002{examples_suffix}"
          ],
          default_rate=1.0)
