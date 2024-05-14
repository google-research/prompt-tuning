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

"""Add T5 and MT5 tasks that include text examples postprocessors/metrics.

This includes several definitions of tasks (mostly from glue and super glue)
using both the T5 and the MT5 sentence piece vocabs. They also include special
postprocessing functions that copy the `inputs_pretokenized` and the
`targets_pretokenized` fields from the input into a dictionary. This dict also
has the model prediction text as well as the model prediction after being
postprocessed. There are also special metric functions, one knows how to
extract the targets and predictions from the dict before running the default
metric functions. There is also a metric function that returns a
`seqio.metrics.Text` object. This is then written to the text tab of
tensorboard.
"""

# pylint: disable=unused-import,g-import-not-at-top
# pytype: disable=import-error
from prompt_tuning.data import c4
from prompt_tuning.data import glue
from prompt_tuning.data import glue_transfer
from prompt_tuning.data import qa
from prompt_tuning.data import summarization
from prompt_tuning.data import super_glue
