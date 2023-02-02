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

"""Special version of the SuperGlue Tasks.

The main task formats here are:

 * super_glue_{name}_v102_examples
 * mt5_super_glue_{name}_v102_examples
 * taskless_super_glue_{name}_v102
 * taskless_super_glue_{name}_v102_examples
 * mt5_taskless_super_glue_{name}_v102
 * mt5_taskless_super_glue_{name}_v102_examples

Any task that starts with `mT5` uses the `mT5` vocab. Any task that ends with
`examples` is setup to log intermediate examples to tensorboard. Any task with
`taskless` does not have the task name as the initial text token (like t5 tasks
do). Any task with `task_index` in the name has a special task index as the
initial post-integerization token.
"""

import functools

from prompt_tuning.data import features
from prompt_tuning.data import metrics as pt_metrics
from prompt_tuning.data import postprocessors as pt_postprocessors
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.data import utils
import seqio
from t5.data import postprocessors
from t5.data import preprocessors
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from t5.evaluation import metrics
import tensorflow_datasets as tfds


super_glue_task_indexer = utils.task_mapping(
    tuple(b.name
          for b in tfds.text.super_glue.SuperGlue.builder_configs.values()), {
              "wsc.fixed": "wsc",
              "axb": "rte",
              "axg": "rte"
          })

for model_prefix, feats in features.MODEL_TO_FEATURES.items():
  for log_examples in (True, False):
    # ========== SuperGlue ==========
    # This section adds the core SuperGlue tasks. We do not include WSC in this
    # loop WSC has different setting for training and validation because t5
    # casts it as a short text generation task instead of as classification (via
    # generation of class labels). We will add that as a mixture later.
    for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
      if "wsc" in b.name:
        continue
      if log_examples:
        postprocess_fn = functools.partial(
            pt_postprocessors.postprocess_with_examples,
            get_glue_postprocess_fn(b))
        metric_fns = [
            functools.partial(pt_metrics.metric_with_examples, func)
            for func in get_super_glue_metric(b.name)
        ] + [functools.partial(pt_metrics.text_examples, task_name=b.name)]
        examples_suffix = "_examples"
      else:
        postprocess_fn = get_glue_postprocess_fn(b)
        metric_fns = get_super_glue_metric(b.name)
        examples_suffix = ""
      # The axb task needs to be rekeyed before we apply the glue text
      # preprocessor, instead of detecting this and registering axb different
      # (which would need to be repeated for each variant of the dataset we
      # have) we have a list of preprocessors, for most tasks this is empty and
      # for axb it has the rekey function. Then when we register a task we add
      # the text processor to this list and it all works out. We can't
      # predefined the full list upfront (like they do in t5) because the actual
      # text preprocessor can be different for tasks like the taskless version.
      pre_preprocessors = []
      if b.name == "axb":
        pre_preprocessors = [
            functools.partial(
                preprocessors.rekey,
                key_map={
                    "premise": "sentence1",
                    "hypothesis": "sentence2",
                    "label": "label",
                    "idx": "idx"
                })
        ]
      # The default tasks have already be register elsewhere so only add the
      # example logging version
      if log_examples:
        seqio.TaskRegistry.add(
            f"{model_prefix}super_glue_{b.name}_v102{examples_suffix}",
            source=seqio.TfdsDataSource(
                tfds_name=f"super_glue/{b.name}:1.0.2",
                splits=["test"] if b.name in ["axb", "axg"] else None),
            preprocessors=pre_preprocessors + [
                get_glue_text_preprocessor(b), seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim
            ],
            postprocess_fn=postprocess_fn,
            metric_fns=metric_fns,
            output_features=feats,
        )
      # This version of the task removes the initial text token of the dataset
      # name
      seqio.TaskRegistry.add(
          f"{model_prefix}taskless_super_glue_{b.name}_v102{examples_suffix}",
          source=seqio.TfdsDataSource(
              tfds_name=f"super_glue/{b.name}:1.0.2",
              splits=["test"] if b.name in ["axb", "axg"] else None),
          preprocessors=pre_preprocessors + [
              get_glue_text_preprocessor(b),
              pt_preprocessors.remove_first_text_token,
              seqio.preprocessors.tokenize,
              seqio.CacheDatasetPlaceholder(),
              seqio.preprocessors.append_eos_after_trim
          ],
          postprocess_fn=postprocess_fn,
          metric_fns=metric_fns,
          output_features=feats,
      )
      # This version of the task adds a task index to the first token.
      seqio.TaskRegistry.add(
          f"{model_prefix}task_index_super_glue_{b.name}_v102{examples_suffix}",
          source=seqio.TfdsDataSource(
              tfds_name=f"super_glue/{b.name}:1.0.2",
              splits=["test"] if b.name in ["axb", "axg"] else None),
          preprocessors=pre_preprocessors + [
              get_glue_text_preprocessor(b),
              pt_preprocessors.remove_first_text_token,
              seqio.preprocessors.tokenize,
              functools.partial(
                  pt_preprocessors.add_sentinel_to_beginning,
                  field="inputs",
                  offset=super_glue_task_indexer[b.name]),
              seqio.CacheDatasetPlaceholder(),
              seqio.preprocessors.append_eos_after_trim
          ],
          postprocess_fn=postprocess_fn,
          metric_fns=metric_fns,
          output_features=feats,
      )

    # ========= Definite Pronoun Resolution =========
    # Similar to the Winograd Schema Challenge but doesn't require semantic
    # knowledge to disambiguate between two different options. Training on this
    # has been shown to be effective for increasing performance on WSC.
    # [Kocijan, et. al., 2019](https://arxiv.org/abs/1905.06290)
    if log_examples:
      dpr_postprocess_fn = functools.partial(
          pt_postprocessors.postprocess_with_examples, utils.identity),
      dpr_metric_fns = [
          functools.partial(pt_metrics.metric_with_examples, metrics.accuracy)
      ] + [functools.partial(pt_metrics.text_examples, task_name="dpr")]
    else:
      dpr_postprocess_fn = utils.identity
      dpr_metric_fns = [metrics.accuracy]

    # DPR without the initial dataset text token.
    seqio.TaskRegistry.add(
        f"{model_prefix}taskless_dpr_v001_simple{examples_suffix}",
        source=seqio.TfdsDataSource(
            tfds_name="definite_pronoun_resolution:1.1.0"),
        preprocessors=[
            preprocessors.definite_pronoun_resolution_simple,
            pt_preprocessors.remove_first_text_token,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=dpr_postprocess_fn,
        metric_fns=dpr_metric_fns,
        output_features=feats,
    )

    seqio.TaskRegistry.add(
        f"{model_prefix}task_index_dpr_v001_simple{examples_suffix}",
        source=seqio.TfdsDataSource(
            tfds_name="definite_pronoun_resolution:1.1.0"),
        preprocessors=[
            preprocessors.definite_pronoun_resolution_simple,
            pt_preprocessors.remove_first_text_token,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            functools.partial(
                pt_preprocessors.add_sentinel_to_beginning,
                field="inputs",
                offset=super_glue_task_indexer["wsc"]),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=dpr_postprocess_fn,
        metric_fns=metric_fns,
        output_features=feats,
    )

    # ========== WSC ==========
    # This adds a "simplified" version of WSC like they do in t5. Instead of
    # predicting if the supplied referent matches the highlighted pronoun in the
    # text, the model generate a referent. If the referent matches the supplied
    # one then the model predictions True, otherwise it will predict false. This
    # means that we can only train on examples where the referent is correct.

    # T5 does WSC in two different tasks. The first is a training task that only
    # uses examples where the referent is true. We never do any evaluation on
    # this dataset so the training data doesn't need anything like post
    # processors or metric_fns. The second task is the evaluation task. This
    # considers all examples and does use the output functions. These tasks are
    # then combined into a mixture.

    # Looking at positive and negative examples of WSC can be hard. If the label
    # is 1 then the target referent should match the models predicted referent.
    # If they match this examples was correct, if they don't the model was
    # wrong. If the label is 0, then the target referent is not correct and we
    # hope the model output something different.
    if log_examples:
      postprocess_fn = functools.partial(
          pt_postprocessors.postprocess_with_examples,
          postprocessors.wsc_simple)
      metric_fns = [
          functools.partial(pt_metrics.metric_with_examples, metrics.accuracy),
          functools.partial(pt_metrics.text_examples, task_name="wsc")
      ]
    else:
      postprocess_fn = postprocessors.wsc_simple
      metric_fns = [metrics.accuracy]

    if log_examples:
      # This version outputs examples to tensorboard.
      seqio.TaskRegistry.add(
          f"{model_prefix}super_glue_wsc_v102_simple_eval{examples_suffix}",
          source=seqio.TfdsDataSource(
              tfds_name="super_glue/wsc.fixed:1.0.2",
              splits=("validation", "test")),
          preprocessors=[
              functools.partial(
                  preprocessors.wsc_simple, correct_referent_only=False),
              seqio.preprocessors.tokenize,
              seqio.CacheDatasetPlaceholder(),
              seqio.preprocessors.append_eos_after_trim,
          ],
          postprocess_fn=postprocess_fn,
          metric_fns=metric_fns,
          output_features=feats)

      # This mixture is WSC where predictions are output to tensorboard.
      seqio.MixtureRegistry.add(
          f"{model_prefix}super_glue_wsc_and_dev_v102_simple{examples_suffix}",
          [
              # We don't need a special version of the training data because it
              # is never processed for output anyway.
              f"{model_prefix}super_glue_wsc_v102_simple_train",
              f"{model_prefix}super_glue_wsc_v102_simple_eval{examples_suffix}"
          ],
          default_rate=1.0)

    # This version remove the initial dataset text token.
    seqio.TaskRegistry.add(
        (f"{model_prefix}taskless_super_glue_wsc_v102_simple_train"
         f"{examples_suffix}"),
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/wsc.fixed:1.0.2", splits=("train",)),
        preprocessors=[
            functools.partial(
                preprocessors.wsc_simple, correct_referent_only=True),
            pt_preprocessors.remove_first_text_token,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[],
        output_features=feats)

    seqio.TaskRegistry.add(
        (f"{model_prefix}taskless_super_glue_wsc_v102_simple_eval"
         f"{examples_suffix}"),
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/wsc.fixed:1.0.2",
            splits=["validation", "test"]),
        preprocessors=[
            functools.partial(
                preprocessors.wsc_simple, correct_referent_only=False),
            pt_preprocessors.remove_first_text_token,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=postprocess_fn,
        metric_fns=metric_fns,
        output_features=feats)

    seqio.MixtureRegistry.add(
        (f"{model_prefix}taskless_super_glue_wsc_and_dev_v102_simple"
         f"{examples_suffix}"),
        [
            # We don't need a special version of the training data because it is
            # never processed for output anyway.
            (f"{model_prefix}taskless_super_glue_wsc_v102_simple_train"
             f"{examples_suffix}"),
            (f"{model_prefix}taskless_super_glue_wsc_v102_simple_eval"
             f"{examples_suffix}")
        ],
        default_rate=1.0)

    # This version adds a task index as the first token.
    seqio.TaskRegistry.add(
        (f"{model_prefix}task_index_super_glue_wsc_v102_simple_train"
         f"{examples_suffix}"),
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/wsc.fixed:1.0.2", splits=("train",)),
        preprocessors=[
            functools.partial(
                preprocessors.wsc_simple, correct_referent_only=True),
            pt_preprocessors.remove_first_text_token,
            seqio.preprocessors.tokenize,
            functools.partial(
                pt_preprocessors.add_sentinel_to_beginning,
                field="inputs",
                offset=super_glue_task_indexer["wsc"]),
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[],
        output_features=feats)

    seqio.TaskRegistry.add(
        (f"{model_prefix}task_index_super_glue_wsc_v102_simple_eval"
         f"{examples_suffix}"),
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/wsc.fixed:1.0.2",
            splits=["validation", "test"]),
        preprocessors=[
            functools.partial(
                preprocessors.wsc_simple, correct_referent_only=False),
            pt_preprocessors.remove_first_text_token,
            seqio.preprocessors.tokenize,
            functools.partial(
                pt_preprocessors.add_sentinel_to_beginning,
                field="inputs",
                offset=super_glue_task_indexer["wsc"]),
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=postprocess_fn,
        metric_fns=metric_fns,
        output_features=feats)

    seqio.MixtureRegistry.add(
        (f"{model_prefix}task_index_super_glue_wsc_and_dev_v102_simple"
         f"{examples_suffix}"),
        [(f"{model_prefix}task_index_super_glue_wsc_v102_simple_train"
          f"{examples_suffix}"),
         (f"{model_prefix}task_index_super_glue_wsc_v102_simple_eval"
          f"{examples_suffix}")],
        default_rate=1.0)


# =========== Mixtures ==========
# These are Mixtures of the task index tasks to train on all super glue tasks
# at once.

# This is a copy of the super glue weights from t5 but adapted to use the task
# index version of the datasets.
WEIGHT_MAPPING = {
    "task_index_super_glue_wsc_v102_simple_train": 259.,
    "task_index_super_glue_wsc_v102_simple_eval_examples": 0.,
    "task_index_super_glue_boolq_v102_examples": 9_427.,
    "task_index_super_glue_cb_v102_examples": 250.,
    "task_index_super_glue_copa_v102_examples": 400.,
    "task_index_super_glue_multirc_v102_examples": 27_243.,
    "task_index_super_glue_record_v102_examples": 138_854.,
    "task_index_super_glue_rte_v102_examples": 2_490.,
    "task_index_super_glue_wic_v102_examples": 5_428.,
}

WEIGHT_MAPPING_WITH_DPR = {
    "task_index_dpr_v001_simple_examples": 1_322.,
    "task_index_super_glue_wsc_v102_simple_train": 259.,
    "task_index_super_glue_wsc_v102_simple_eval_examples": 0.,
    "task_index_super_glue_boolq_v102_examples": 9_427.,
    "task_index_super_glue_cb_v102_examples": 250.,
    "task_index_super_glue_copa_v102_examples": 400.,
    "task_index_super_glue_multirc_v102_examples": 27_243.,
    "task_index_super_glue_record_v102_examples": 138_854.,
    "task_index_super_glue_rte_v102_examples": 2_490.,
    "task_index_super_glue_wic_v102_examples": 5_428.,
}

seqio.MixtureRegistry.add("task_index_super_glue_v102_examples_proportional",
                          list(WEIGHT_MAPPING.items()))

seqio.MixtureRegistry.add(
    "task_index_super_glue_with_dpr_v102_examples_proportional",
    list(WEIGHT_MAPPING_WITH_DPR.items()))
