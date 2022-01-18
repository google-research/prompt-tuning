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

r"""Convert the MRQA data format into a tsv format that t5 can handle.

This script converts the jsonl format from MRQA into a tsv file for t5
consumption. It produces a tsv with the following columns. `id`, `question`,
`context`, `answer` and `answers`. `answer` is the answer that will be used as
the target when training while `answers` will be the multiple human answers
that are used for evaluation. The answers column will be joined together with
the `--delim` argument (defaults to `"|||"`).

Eventually it should be nice to have a tfds version of these datasets
(at least the out of domain validation set ones) that produce a
`squad-like` example format. There are some hurdles for that though.
Are all the answers actually in the text? Do all the answers have start
offsets associated with them?

Example usage:
python -m prompt_tuning.scripts.mrqa_to_tsv \
    --json /path/to/MRQA_dataset.jsonl \
    --output /path/to/MRQA_dataset.tsv

"""

import json
import os
import re
from typing import Dict

from absl import app
from absl import flags
import pandas as pd
from tensorflow.io import gfile

FLAGS = flags.FLAGS
flags.DEFINE_string("jsonl", None, "The MRQA 2019 shared task jsonl file.")
flags.DEFINE_integer("header", 1, "The number of header lines to skip.")
flags.DEFINE_string(
    "delim", "|||",
    "The default separator of answers when we serialize the list")
flags.DEFINE_string(
    "output", None,
    "The name of the output file. Defaults to `--jsonl` base name + \".tsv\"")
flags.mark_flag_as_required("jsonl")


def normalize_whitespace(s: str) -> str:
  """Convert all whitespace (tabs, newlines, etc) into spaces."""
  return re.sub(r"\s+", " ", s, flags=re.MULTILINE)


def parse_line(line: str, delim: str = "|||") -> Dict[str, str]:
  """Turn a jsonl line into a row that is ready to write to csv."""
  example = json.loads(line)
  context = normalize_whitespace(example["context"])
  # Some of the questions have newlines in them and the tensorflow csv utils
  # can't handle newlines, so remove them.
  question = normalize_whitespace(example["qas"][0]["question"])
  qid = example["qas"][0]["qid"]
  answer = normalize_whitespace(
      example["qas"][0]["detected_answers"][0]["text"])
  answers = [normalize_whitespace(ans) for ans in example["qas"][0]["answers"]]
  return {
      "id": qid,
      "context": context,
      "question": question,
      "answer": answer,
      "answers": delim.join(answers)
  }


def main(_):
  output = FLAGS.output
  if FLAGS.output is None:
    output = os.path.splitext(FLAGS.jsonl)[0] + ".tsv"

  examples = []
  with gfile.GFile(FLAGS.jsonl) as f:
    for _ in range(FLAGS.header):
      f.readline()
    for line in f:
      examples.append(parse_line(line))

  df = pd.DataFrame(examples, dtype=str)
  with gfile.GFile(output, "w") as wf:
    df.to_csv(wf, index=False, sep="\t")


if __name__ == "__main__":
  app.run(main)
