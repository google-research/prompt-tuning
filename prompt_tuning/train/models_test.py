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

"""Tests for models."""

import functools
from unittest import mock
from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from prompt_tuning.train import models
from t5x import decoding


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

  def test_predict_batch_returns_best(self):
    batch = {
        "decoder_input_tokens": np.array([
            [0, 3, 4, 5, 6, 0, 0],
        ]),
        "decoder_causal_attention": np.array([
            [1, 1, 1, 0, 0, 0, 0],
        ])
    }

    # These dummy logits represent the probability distribution where all the
    # probability mass is in one item (i.e., degenerate distribution). For
    # batch element 0, it is vocabulary index 2. We have two samples.
    # Technically these should be identical since the prompts are the same, but
    # this makes testing easier.
    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, -1]]), axis=1)

    mock_module = mock.Mock()
    mock_module.apply.return_value = (dummy_logits, {"cache": {}})
    mock_module.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_module
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = functools.partial(decoding.temperature_sample, topk=4)
      self._inputs_bidirectional_attention = False
      self.prompt_length = 10

    with mock.patch.object(
        models.PromptDecoderOnlyModel, "__init__", new=mock_init):
      model = models.PromptDecoderOnlyModel()

    actual_output, aux = model.predict_batch_with_aux({},
                                                      batch,
                                                      num_decodes=2,
                                                      return_all_decodes=False)

    expected_output = [[3, 3, 3, 3, 3, 0, 0]]
    expected_scores = [0.]

    # The expected progression of the first element of 'decoder_input_tokens':
    # [0, 3, 4, 5, 6, 0, 0] -> [0, 3, 4, 0, 0, 0, 0] ->
    # [3, 4, 2, 2, 2, 2, 2] -> [2, 2, 2, 2, 2, 0, 0]

    # The expected progression of the second element of 'decoder_input_tokens':
    # [0, 7, 8, 9, 0, 0, 0] -> [0, 7, 0, 0, 0, 0, 0] ->
    # [7, 3, 3, 3, 3, 3, 3] -> [3, 3, 3, 3, 3, 3, 0]

    np.testing.assert_array_equal(actual_output, expected_output)
    np.testing.assert_array_equal(aux["scores"], expected_scores)

  def test_predict_batch_all_decoders(self):
    batch = {
        "decoder_input_tokens": np.array([
            [0, 3, 4, 5, 6, 0, 0],
        ]),
        "decoder_causal_attention": np.array([
            [1, 1, 1, 0, 0, 0, 0],
        ])
    }

    # These dummy logits represent the probability distribution where all the
    # probability mass is in one item (i.e., degenerate distribution). For
    # batch element 0, it is vocabulary index 2. We have two samples.
    # Technically these should be identical since the prompts are the same, but
    # this makes testing easier.
    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]]), axis=1)

    mock_module = mock.Mock()
    mock_module.apply.return_value = (dummy_logits, {"cache": {}})
    mock_module.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_module
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = functools.partial(decoding.temperature_sample, topk=4)
      self._inputs_bidirectional_attention = False
      self.prompt_length = 10

    with mock.patch.object(
        models.PromptDecoderOnlyModel, "__init__", new=mock_init):
      model = models.PromptDecoderOnlyModel()

    actual_output, aux = model.predict_batch_with_aux({},
                                                      batch,
                                                      num_decodes=2,
                                                      return_all_decodes=True)

    expected_output = [[[2, 2, 2, 2, 2, 0, 0], [3, 3, 3, 3, 3, 0, 0]]]
    expected_scores = [[0., 0.]]

    # The expected progression of the first element of 'decoder_input_tokens':
    # [0, 3, 4, 5, 6, 0, 0] -> [0, 3, 4, 0, 0, 0, 0] ->
    # [3, 4, 2, 2, 2, 2, 2] -> [2, 2, 2, 2, 2, 0, 0]

    # The expected progression of the second element of 'decoder_input_tokens':
    # [0, 7, 8, 9, 0, 0, 0] -> [0, 7, 0, 0, 0, 0, 0] ->
    # [7, 3, 3, 3, 3, 3, 3] -> [3, 3, 3, 3, 3, 3, 0]

    np.testing.assert_array_equal(actual_output, expected_output)
    np.testing.assert_array_equal(aux["scores"], expected_scores)


if __name__ == "__main__":
  absltest.main()
