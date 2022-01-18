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

"""Tests for multitask training prompts."""

from unittest import mock
from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from prompt_tuning.extended import multitask_prompts
from prompt_tuning.extended.train import multitask_prompts as train_multitask_prompts
from prompt_tuning.train import prompts as train_prompts


class PromptsTest(absltest.TestCase):

  def test_multitask_prompt_does_concatenation(self):
    embed_size = 20
    prompt_length = 5
    batch_size = 2
    seq_len = 20
    mock_prompt = mock.create_autospec(
        multitask_prompts.MultiTaskPrompt, spec_set=True, instance=True)
    prompt = jnp.zeros((batch_size, prompt_length, embed_size))
    mock_prompt.return_value = prompt
    mock_combine = mock.create_autospec(
        train_prompts.prefix_prompt, spec_set=True)
    prompt_module = train_multitask_prompts.MultiTaskPrompt(
        prompt=mock_prompt, combine=mock_combine)
    input_tokens = jnp.ones((batch_size, seq_len))
    embed = jnp.ones((batch_size, seq_len, embed_size))
    prompt_module.apply({"params": {}}, input_tokens, embed)
    self.assertEqual(mock_prompt.call_args_list[0],
                     mock.call(input_tokens, embed))
    np.testing.assert_allclose(mock_combine.call_args_list[0][0][0], prompt)
    np.testing.assert_allclose(mock_combine.call_args_list[0][0][1], embed[:,
                                                                           1:])


if __name__ == "__main__":
  absltest.main()
