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

"""T5X model subclasses for prompt tuning."""

import functools
from typing import Optional, Mapping, MutableMapping, Any, Tuple
import jax
import jax.numpy as jnp
from t5x import models
from flaxformer.types import Array


PyTreeDef = type(jax.tree_structure(None))


class PromptDecoderOnlyModel(models.DecoderOnlyModel):
  """A prompted DecoderOnly Model that uses the prefill cache for prompting."""

  def __init__(self,
               prompt_length: int,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.prompt_length = prompt_length

  def _compute_logits(self,
                      params: Mapping[str, Array],
                      batch: Mapping[str, jnp.ndarray],
                      dropout_rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Hack off the prompt when calculating logits."""
    logits = super()._compute_logits(params, batch, dropout_rng)
    return logits[:, self.prompt_length:]

  def predict_batch_with_aux(
      self,
      params: Mapping[str, Array],
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.random.KeyArray] = None,
      num_decodes: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    # Get the maximum shape of an example, [B, input_tokens + target_tokens]
    target_shape = batch['decoder_input_tokens'].shape
    target_type = batch['decoder_input_tokens'].dtype
    # We need this to be the full cache (including the prompt) length so masks
    # and biases match when used with single slices.
    max_decode_length = target_shape[1] + self.prompt_length

    # Initialize a zero'd out auto-regressive cache. The shape for the keys and
    # values are [B, ..., max_decode_length], the cache_index is a scalar, and
    # the relpos_bias cache is
    # [1, num_heads, max_decode_length, max_decode_length].
    #
    # We need to call this with `decode=True` so that the relpos cache is
    # created, otherwise it would get created inside of the `lax.while_loop` and
    # cause an error. However, when the `layers.PromptDecoderOnly` is called
    # with `decode=True` the prompt is not added to the input because the input
    # is expected to have a single step in the sequence dimension (The fact that
    # this input has more than 1 step is okay because the single step check in
    # the dense attention class is after the `is_initialized` check. When this
    # is called with more than one step, the cache is created but the
    # `is_initialized` check fails so we never get to the shape check).
    #
    # We can't set `prefill=True` which would normally be used to make sure the
    # prompt is applied to the inputs, because then both `prefill` and `decode`
    # are true and this causes an error (eventually `prefill` should be
    # refactored to be a state of `decode` but that is too much to change right
    # now). This means that our cache would only the input shape, there
    # wouldn't be room for the prompt. So we update the shape of the dummy data
    # to include the prompt.
    #
    # Now we can call apply with `decode=True` (so all parts of the cache are
    # initialized) and the shapes will be correct (they will be longer to fit
    # the prompts).
    prompted_shape = target_shape[:1] + (max_decode_length,)
    _, variables_with_cache = self.module.apply(
        {'params': params},
        jnp.ones(prompted_shape, target_type),
        jnp.ones(prompted_shape, target_type),
        enable_dropout=False,
        decode=True,
        prefill=False,
        mutable=['cache'])
    cache = variables_with_cache['cache']

    # Calculate the size of the inputs for each example by summing their
    # causal attention mask. This mask has 1 extra 1 token than the number of
    # inputs so subtract off 1.
    inputs_lengths = jnp.sum(batch['decoder_causal_attention'], axis=1) - 1

    # Since decoder_input_tokens is shifted to the right AND
    # `decoder_causal_attention` has one more 1 than the number of inputs
    # tokens, this sifts out targets portion of the decoder_input_tokens.
    inputs = batch['decoder_input_tokens'] * batch['decoder_causal_attention']

    # We can prefill the cache with both the prompt representations and the
    # input representations so our prefill length is the inputs plus the prompt
    # length.
    prefill_lengths = inputs_lengths + self.prompt_length

    # If `self._inputs_bidirectional_attention = False`, we should not pass
    # batch['decoder_causal_attention'] to `module.apply` during cache prefill
    # and pass None instead.
    maybe_decoder_causal_attention = self._get_decoder_causal_attention(batch)

    # This prefills the cache. Because prefill=True, the
    # layers.PromptDecoderOnly will add the prompt to the input.
    _, variables_with_cache = self.module.apply(
        {
            'params': params,
            'cache': cache,
        },
        inputs,
        # Use the `decoder_causal_attention` as the targets so the decoder
        # attention mask will correctly cover the whole input, otherwise the
        # first input token (the 0 for BOS) will not be included in the mask.
        # This also restricts the mask to not include any target positions like
        # it would if you used `decoder_target_tokens`.
        batch['decoder_causal_attention'],
        decoder_causal_attention=maybe_decoder_causal_attention,
        mutable=['cache'],
        enable_dropout=False,
        prefill=True,
        prefill_lengths=prefill_lengths,
    )
    # The cache index will now be converted to a vector of `[B]`
    prefilled_cache = variables_with_cache['cache']

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        max_decode_length=max_decode_length)

    # Make sure that `decoder_params` can be unpacked with `**decoder_params`
    # below.
    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # When we run the actual decode function we will be indexing into the input
    # array to extract the next token (or write it when generating). It make
    # sure this index (which was created from prefilling and includes the prompt
    # length) is the correct location we need to pad the input as if there were
    # prompts added.

    # Note: Adding the prompt to the end isn't well-formed. If the prompt
    # (which doesn't have a token) is the last position, what should the last
    # input token (which is fed into the decode function first) be? To insert
    # a prompt at the "end" of the input we should actually insert it just
    # before EOS. Decoding will begin part way though the inputs, the actual
    # prompt positions will never be used as token inputs so their value does
    # not matter (although, we use 2 just to avoid any complications multiple
    # EOS tokens might bring), additionally, because the cached keys and value
    # are used, we don't need the prompt to be inserted in the real location, we
    # just need the shift in the location of the EOS to change appropriately.
    # Thus we can use this padding with experiments where the prompt is not
    # prepended.
    prompt_pad = jnp.full((inputs.shape[0], self.prompt_length), 2,
                          dtype=inputs.dtype)
    inputs_with_fake_prompts = jnp.concatenate([prompt_pad, inputs], axis=1)

    # Using the above-defined single-step decoder function, run a decoding
    # function that will produces a [batch, 1, max_decode_length] output.
    decoded_sequences, scores = self._decode_fn(
        inputs=inputs_with_fake_prompts,
        cache=prefilled_cache,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=self.output_vocabulary.eos_id,
        num_decodes=num_decodes,
        initial_index=prefill_lengths,
        **decoder_params)

    # Remove the (possibly fake) beam dimension.
    decoded_sequences = decoded_sequences[:, -1, :]

    # Beam search returns []
    aux = {'scores': scores[:, -1]}

    # Return the decoded sequences with an empty aux output dict.
    sequences = models.remove_prefix(decoded_sequences, prefill_lengths)
    # Remove the extra space the prompt took up (these are all shifted to the
    # end and zero'd out).
    trimmed_sequences = sequences[:, :-self.prompt_length]
    return trimmed_sequences, aux
