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

r"""Recycle a single prompt.

"""

from absl import app
from absl import flags
from absl import logging
from prompt_tuning import prompts
from prompt_tuning.recycling import recycle


_SOURCE_MODEL = flags.DEFINE_string(
    "source_model",
    "",
    "The Source Model the prompt with trained with.")
_TARGET_MODEL = flags.DEFINE_string(
    "target_model",
    "",
    "The Target Model we want to recycle to.")
_PROMPT_PATH = flags.DEFINE_string(
    "prompt_path",
    "",
    "The prompt (trained with --source_model) you want to recycle")
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "",
    "Where to save the recycled prompt.")
_FILTERED_WORDS = flags.DEFINE_string(
    "filtered_word",
    "prompt_tuning/recycling/data/filtered-vocab-english-only.json",
    "The list of words to load from the vocab")
_NUM_WORDS = flags.DEFINE_integer(
    "num_words",
    4000,
    "The number of words to use for training the recycler.")
_WORD_OFFSET = flags.DEFINE_integer(
    "word_offset",
    1000,
    "The number of words to skip before training on --num_words.")
_RECYCLER = flags.DEFINE_enum(
    "recycler",
    "v2v-nn",
    ["v2v-nn", "v2v-lin", "lin-comb"],
    "The Recycler to use.")
_RECYCLER_SCALE = flags.DEFINE_integer(
    "hidden_scale",
    4,
    "Hidden size scaling factor, only used for --recycler=v2v-nn")
_RECYCLER_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    50,
    "Batch size used for training, only used for --recycler=v2v-nn")
_RECYCLER_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    0.0003,
    "Learning rate to use when training, only used for --recycler=v2v-nn")
_RECYCLER_EPOCHS = flags.DEFINE_integer(
    "epochs",
    25,
    "Number of epochs to train the recycler, only used for --recycler=v2v-nn")

flags.mark_flags_as_required(
    ["source_model", "target_model", "prompt_path", "output_path"])


def translate_recycler_name(name: str) -> str:
  name_ = name.lower()
  if name_ == "v2v-nn":
    return "jax-nn"
  elif name_ == "v2v-lin":
    return "tf-lstsq"
  elif name_ == "lin-comb":
    return "linear-combination"
  else:
    raise ValueError(f"Didn't understand --recycler={name}")


def main(argv):
  """Run a single Recycler."""
  del argv

  emb_config = {"load_embeddings": {"default": {
      "word_list_path": _FILTERED_WORDS.value,
      "num_words": _NUM_WORDS.value,
      "word_offset": _WORD_OFFSET.value
  }}}
  logging.info(
      "Loading Source Model embeddings from %s using the list from %s to load "
      "%d after skipping %d",
      _SOURCE_MODEL.value,
      _FILTERED_WORDS.value,
      _NUM_WORDS.value,
      _WORD_OFFSET.value)
  source_embeddings = recycle.load_embeddings(
      _SOURCE_MODEL.value,
      "default",
      emb_config)
  logging.info(
      "Loading Target Model embeddings from %s using the list from %s to load "
      "%d after skipping %d",
      _TARGET_MODEL.value,
      _FILTERED_WORDS.value,
      _NUM_WORDS.value,
      _WORD_OFFSET.value)
  target_embeddings = recycle.load_embeddings(
      _TARGET_MODEL.value,
      "default",
      emb_config)

  recycler_config = {
      "recycling_methods": {
          "jax-nn": {
              "__init__": {
                  "hidden_scale": _RECYCLER_SCALE.value
              }
          },
          "linear-combination": {},
          "tf-lstsq": {}}}
  logging.info("Creating the %s recycler.", _RECYCLER.value)
  if _RECYCLER.value == "v2v-nn":
    logging.info(
        "Hidden size is the target embeddings size (%d) times (%d) = (%d)",
        target_embeddings.shape[-1],
        _RECYCLER_SCALE.value,
        target_embeddings.shape[-1] * _RECYCLER_SCALE.value)
  recycler = recycle.make_recycler(
      translate_recycler_name(_RECYCLER.value),
      recycler_config,
      source_embeddings,
      target_embeddings)

  recycler_fit_config = {
      "batch_size": _RECYCLER_BATCH_SIZE.value,
      "learning_rate": _RECYCLER_LEARNING_RATE.value,
      "epochs": _RECYCLER_EPOCHS.value
  }
  logging.info("Fitting recycler.")
  if _RECYCLER.value == "v2v-nn":
    logging.info(
        "Training with batch_size=%d and learning_rate=%f for %d epochs.",
        _RECYCLER_BATCH_SIZE.value,
        _RECYCLER_LEARNING_RATE.value,
        _RECYCLER_EPOCHS.value)
  recycler = recycler.fit(
      source_embeddings,
      target_embeddings,
      **recycler_fit_config)

  source_prompt = prompts.np_load(_PROMPT_PATH.value)
  logging.info(
      "Loading Source Prompt (shape=%s) from %s",
      source_prompt.shape,
      _PROMPT_PATH.value)

  logging.info("Recycling Prompt.")
  recycled_prompt = recycler(source_prompt)

  logging.info(
      "Saving Recycled Prompt (shape=%s) to %s",
      recycled_prompt.shape,
      _OUTPUT_PATH.value)
  recycle.np_save(_OUTPUT_PATH.value, recycled_prompt)


if __name__ == "__main__":
  app.run(main)
