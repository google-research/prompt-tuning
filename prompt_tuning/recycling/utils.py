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

"""Utilities for reading english only embeddings."""

import json
from t5x import checkpoints
from tensorflow.io import gfile

from flaxformer.types import Array


def load_english_only_embedding(
    checkpoint_path: str,
    word_list_path: str,
) -> Array:
  """Load embedding based on our filtered, english-only list.

  Args:
    checkpoint_path: The t5x checkpoint to load from.
    word_list_path: The word list json file, format is Mapping[int, str] where
      the key is the vocab index in the original vocab and the value is the
      actual text of the word.

  Returns:
    The embedding table for the words in `word_list_path`.
  """
  with gfile.GFile(word_list_path) as f:
    filtered_words = json.load(f)

  # JSON keys are strings, convert to ints for actual indexing.
  filtered_indices = list(map(int, filtered_words.keys()))

  full_embeddings = checkpoints.load_t5x_checkpoint(
      checkpoint_path, lazy_parameters=True)
  full_embeddings = (
      full_embeddings["target"]["token_embedder"]["embedding"].get())
  return full_embeddings[filtered_indices]
