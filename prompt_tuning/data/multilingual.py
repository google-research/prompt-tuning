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

# ----- WikiLingua -----
# WikiLingua zero-shot task. The train split consists only of English data and
# evaluation is done on non-English data. The base evaluation is input:
# non-EN article and target: EN summary.

# Dictionary mapping language code to language string used in the WikiLingua
# task definition.
WIKILINGUA_LANGS = {
    'ar': 'arabic_ar',
    'zh': 'chinese_zh',
    'cs': 'czech_cs',
    'nl': 'dutch_nl',
    'en': 'english_en',
    'fr': 'french_fr',
    'de': 'german_de',
    'hi': 'hindi_hi',
    'id': 'indonesian_id',
    'it': 'italian_it',
    'ja': 'japanese_ja',
    'ko': 'korean_ko',
    'pt': 'portuguese_pt',
    'ru': 'russian_ru',
    'es': 'spanish_es',
    'th': 'thai_th',
    'tr': 'turkish_tr',
    'vi': 'vietnamese_vi'
}

for model_prefix, model_features in features.MODEL_TO_FEATURES.items():
  # English train.
  t5.data.TaskRegistry.add(
      f'{model_prefix}pt_wikilingua.english_en_train',
      t5.data.TfdsTask,
      tfds_name='gem/wiki_lingua_english_en:1.1.0',
      splits=['train'],
      text_preprocessor=[
          functools.partial(
              pt_preprocessors.preprocess_wikilingua,
              lang='en')
      ],
      output_features=model_features,
      metric_fns=[metrics.rouge])

  # Evaluation on other languages.
  for lang_code, lang in WIKILINGUA_LANGS.items():
    t5.data.TaskRegistry.add(
        f'{model_prefix}pt_wikilingua_validation.{lang_code}',
        t5.data.TfdsTask,
        tfds_name=f'gem/wiki_lingua_{lang}:1.1.0',
        splits={'validation': 'validation'},
        text_preprocessor=[
            functools.partial(
                pt_preprocessors.preprocess_wikilingua,
                lang=lang_code)
        ],
        output_features=model_features,
        metric_fns=[metrics.rouge])

  t5.data.MixtureRegistry.add(
      f'{model_prefix}pt_wikilingua_zeroshot',
      ([f'{model_prefix}pt_wikilingua.english_en_train'] +
       [f'{model_prefix}pt_wikilingua_validation.{lang_code}'
        for lang_code in WIKILINGUA_LANGS.keys()]),
      default_rate=1.0)


# ----- mC4 -----
# Left-to-Right Language Modeling: This task has an empty input and text in the
# targets. This can be used as the input to a DecoderOnly model, like normal, or
# it can be used as the input to an EncoderDecoder Model, in which case the
# inputs will be only a single EOS token. This is akin to just using the decoder
# of an EncoderDecoder model as a DecoderOnly model.
#
# We expect this task to be useful for learning "language prompts" that steer
# the model toward a specific output language as the only thing the decoder will
# be able to condition on is the prompt.
#
# With the addition of the language filter, training the mixture may fail due
# to memory issues. Training individual language prompts as separate tasks is
# recommended to avoid these issues with memory.
for lang in tfds.text.c4.MC4_LANGUAGES:
  seqio.TaskRegistry.add(
      f'mc4_lm.{lang.replace("-", "_")}',
      source=seqio.TfdsDataSource(
          tfds_name='c4/multilingual:3.0.1',
          splits={
              'train': lang,
              'validation': f'{lang}-validation',
          }
      ),
      preprocessors=[
          functools.partial(
              pt_preprocessors.filter_langid,
              lang_code=lang,
              lang_detector=_lang_detector),
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  'inputs': None,
                  'targets': 'text'
              }
          ),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=None,
      metric_fns=[],
      output_features=features.MT5_FEATURES
  )

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE
)

seqio.MixtureRegistry.add(
    'mc4_lm',
    [f'mc4_lm.{lang.replace("-", "_")}'
     for lang in tfds.text.c4.MC4_LANGUAGES],
    default_rate=DEFAULT_MIX_RATE
)
