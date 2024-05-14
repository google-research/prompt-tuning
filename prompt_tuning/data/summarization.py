# Copyright 2024 Google.
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

"""Summarization tasks for T5.

This includes some summarization tasks, xsum and cnn/dailymail, as t5 tasks
expect they are configured to write examples to Tensorboard when they run
evaluation.
"""
import functools

from prompt_tuning.data import features
from prompt_tuning.data import metrics as rom_metrics
from prompt_tuning.data import postprocessors as rom_postprocessors
from prompt_tuning.data import utils
import seqio
from t5.data import preprocessors
from t5.evaluation import metrics


# ========== Summarization ==========
for model_prefix, feats in features.MODEL_TO_FEATURES.items():
  # ===== XSUM =====
  seqio.TaskRegistry.add(
      f"{model_prefix}xsum_v110_examples",
      source=seqio.TfdsDataSource(tfds_name="xsum:1.1.0"),
      preprocessors=[
          functools.partial(
              preprocessors.summarize,
              article_key="document",
              summary_key="summary"),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          rom_postprocessors.postprocess_with_examples, utils.identity),
      metric_fns=[
          functools.partial(rom_metrics.metric_with_examples, metrics.rouge),
          functools.partial(
              rom_metrics.text_examples, task_name="xsum", find_negative=False),
      ],
      output_features=feats)

  # ===== CNN/DailyMail =====
  seqio.TaskRegistry.add(
      f"{model_prefix}cnn_dailymail_v310_examples",
      source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.1.0"),
      preprocessors=[
          functools.partial(
              preprocessors.summarize,
              article_key="article",
              summary_key="highlights"),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          rom_postprocessors.postprocess_with_examples, utils.identity),
      metric_fns=[
          functools.partial(rom_metrics.metric_with_examples, metrics.rouge),
          functools.partial(
              rom_metrics.text_examples,
              task_name="cnn_dailymail",
              find_negative=False)
      ],
      output_features=feats)
