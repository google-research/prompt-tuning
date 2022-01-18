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

"""Tests for models."""

from unittest import mock
from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from prompt_tuning.train import models


class PromptPrefixDecoderOnlyModelTest(absltest.TestCase):

  @mock.patch.object(
      models.models.DecoderOnlyModel, "_compute_logits", autospec=True)
  def test_compute_logits_removes_prompt(self, logits_mock):
    prompt_length = 20
    logits_length = 54
    batch_size = 5
    logits = jnp.ones((batch_size, logits_length))
    prompt = jnp.zeros((batch_size, prompt_length))
    logits_with_prompt = jnp.concatenate([prompt, logits], axis=1)

    def mock_init(self):
      self.prompt_length = prompt_length

    with mock.patch.object(
        models.PromptDecoderOnlyModel, "__init__", new=mock_init):
      prompt_prefix_lm = models.PromptDecoderOnlyModel()

    logits_mock.return_value = logits_with_prompt
    results = prompt_prefix_lm._compute_logits({"params": {}}, {}, None)

    self.assertEqual(results.shape, (batch_size, logits_length))
    np.testing.assert_allclose(results, logits)


if __name__ == "__main__":
  absltest.main()
