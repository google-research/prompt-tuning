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

"""Custom adafactor rules.
"""

from flax.core.frozen_dict import freeze
from flax.core.frozen_dict import unfreeze
from t5x import adafactor


def standard_logical_factor_rules(rules=None):
  """Add prompt adafactor rules to your set of rules."""
  if rules is None:
    rules = adafactor.standard_logical_factor_rules()
  rules = unfreeze(rules)
  rules['prompt'] = adafactor.FactorDim.NONE
  rules['tasks'] = adafactor.FactorDim.NONE
  rules['prompt+embed'] = adafactor.FactorDim.NONE
  rules['prompt_embed'] = adafactor.FactorDim.NONE
  rules['batch'] = adafactor.FactorDim.BATCH
  return freeze(rules)
