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

"""Tasks definitions for multilingual prompt tuning."""

import functools
import threading

from multilingual_t5 import preprocessors
from multilingual_t5 import utils
from prompt_tuning.data import features
from prompt_tuning.data import preprocessors as pt_preprocessors
import seqio
import t5.data
from t5.evaluation import metrics
import tensorflow_datasets as tfds


DEFAULT_SPM_PATH = 'gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model'

# Global lock used for language detection modules that aren't threadsafe.
langdetect_lock = threading.Lock()


class LangDetector:
  """A Language ID detector class."""

  def __init__(self):
    with langdetect_lock:
      self._detector = tfds.core.lazy_imports.gcld3.NNetLanguageIdentifier(
          # CLD3 is not expected to work well on very short documents.
          min_num_bytes=100,
          max_num_bytes=10000)

  def find_language(self, text):
    """Detects the language of the text."""
    with langdetect_lock:
      return self._detector.FindLanguage(text=text)

_lang_detector = LangDetector()


# ----- XNLI -----
# XNLI zero-shot task. This trains on English MNLI training data and then
# evaluates on multilingual XNLI dev/test data.

XNLI_LANGS = [
    'ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr',
    'ur', 'vi', 'zh'
]

for model_prefix, model_features in features.MODEL_TO_FEATURES.items():
  seqio.TaskRegistry.add(
      f'{model_prefix}pt_xnli_train',
      source=seqio.TfdsDataSource(
          tfds_name='multi_nli:1.1.0', splits=['train']),
      preprocessors=[
          preprocessors.process_mnli,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=model_features,
      metric_fns=[metrics.accuracy])

  for split in ['validation', 'test']:
    for lang in XNLI_LANGS:
      seqio.TaskRegistry.add(
          f'{model_prefix}pt_xnli_{split}.{lang}',
          # Note that both the test and validation splits are defined under the
          # 'validation' split. This is necessary because T5X can take only
          # one test / validation split as of now. The task name contains the
          # actual split.
          source=seqio.TfdsDataSource(
              tfds_name='xnli:1.1.0', splits={'validation': f'{split}'}),
          preprocessors=[
              functools.partial(
                  preprocessors.process_xnli, target_languages=[lang]),
              seqio.preprocessors.tokenize,
              seqio.CacheDatasetPlaceholder(),
              seqio.preprocessors.append_eos_after_trim,
          ],
          output_features=model_features,
          metric_fns=[metrics.accuracy])

    seqio.TaskRegistry.add(
        f'{model_prefix}pt_xnli_{split}.all_langs',
        source=seqio.TfdsDataSource(
            tfds_name='xnli:1.1.0', splits={'validation': f'{split}'}),
        preprocessors=[
            functools.partial(
                preprocessors.process_xnli, target_languages=XNLI_LANGS),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=model_features,
        metric_fns=[metrics.accuracy])

  xnli_zeroshot = [f'{model_prefix}pt_xnli_train']
  for split in ['validation', 'test']:
    xnli_zeroshot += ([f'{model_prefix}pt_xnli_{split}.all_langs'] +
                      [f'{model_prefix}pt_xnli_{split}.{lang}'
                       for lang in XNLI_LANGS])

  seqio.MixtureRegistry.add(
      f'{model_prefix}pt_xnli_zeroshot', xnli_zeroshot, default_rate=1.0)

# ----- XQuAD -----
# XQuAD zero-shot task. This task uses the train split from English SQuAD,
# the validation split from English SQuAD (to identify the best checkpoint)
# and then evaluates on multilingual XQuAD.
for model_prefix, model_features in features.MODEL_TO_FEATURES.items():
  seqio.TaskRegistry.add(
      f'{model_prefix}pt_squad_train',
      source=seqio.TfdsDataSource(
          tfds_name='squad/v1.1:3.0.0', splits=['train']),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=model_features,
      metric_fns=[metrics.squad])

  # SQuAD validation.
  seqio.TaskRegistry.add(
      f'{model_prefix}pt_squad_validation',
      source=seqio.TfdsDataSource(
          tfds_name='squad/v1.1:3.0.0', splits=['validation']),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=model_features,
      metric_fns=[metrics.squad])

  # XQuAD test.
  for lang in utils.XQUAD_LANGS_TEST:
    seqio.TaskRegistry.add(
        f'{model_prefix}pt_xquad_test.{lang}',
        # Note that the validation split is mapped to the XQuAD 'test' split
        # because T5X can only take in one validation / test split.
        source=seqio.TfdsDataSource(
            tfds_name=f'xquad/{lang}:3.0.0', splits={'validation': 'test'}),
        preprocessors=[
            preprocessors.xquad,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=model_features,
        metric_fns=[metrics.squad])

  # Additional test task containing all the languages.
  seqio.TaskRegistry.add(
      f'{model_prefix}pt_xquad_test.all_langs',
      source=seqio.FunctionDataSource(
          dataset_fn=utils.xquad_all_langs_dataset_fn,
          splits={'validation': 'test'}),
      preprocessors=[
          preprocessors.xquad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5.data.postprocessors.qa,
      output_features=model_features,
      metric_fns=[metrics.squad])

  # XQuAD Zero-Shot (SQuAD train, SQuAD dev, XQuAD test).
  xquad_test = ([f'{model_prefix}pt_xquad_test.{lang}'
                 for lang in utils.XQUAD_LANGS_TEST] +
                [f'{model_prefix}pt_xquad_test.all_langs'])

  xquad_zeroshot = ([f'{model_prefix}pt_squad_train',
                     f'{model_prefix}pt_squad_validation'] +
                    xquad_test)

  seqio.MixtureRegistry.add(
      f'{model_prefix}pt_xquad_zeroshot', xquad_zeroshot, default_rate=1.0)

# ----- GEM-XSum -----
# XSum is an English language only summarization task where the inputs are
# articles from the BBC and the targets are one line summaries of the articles.
_rouge_fn = functools.partial(
    metrics.rouge,
    score_keys=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])

for model_prefix, model_features in features.MODEL_TO_FEATURES.items():
  seqio.TaskRegistry.add(
      f'{model_prefix}pt_gem_xsum',
      source=seqio.TfdsDataSource(tfds_name='gem/xsum:1.0.1'),
      preprocessors=[
          functools.partial(
              t5.data.preprocessors.summarize,
              article_key='document',
              summary_key='target'),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[metrics.bleu, _rouge_fn],
      output_features=model_features)

