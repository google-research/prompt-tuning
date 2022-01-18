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

r"""Find where a module is installed.

This tool is useful for finding where a package like T5X is installed so that
we can easily use the gin configs that are bundled with it.

Example usage:

python -m t5x.train \
  --gin_search_paths=`python -m prompt_tuning.scripts.find_module t5x` \
  --gin_file=t5x/configs/... \
  ...
"""

import importlib
import os
from typing import Sequence
from absl import app


def main(argv: Sequence[str]):
  if len(argv) != 2:
    raise app.UsageError("Missing module argument.")

  module = importlib.import_module(argv[1])
  print(os.path.dirname(os.path.abspath(module.__file__)))


if __name__ == "__main__":
  app.run(main)
