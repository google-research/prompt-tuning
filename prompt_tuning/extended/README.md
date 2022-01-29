# Extensions to Prompt Tuning

This directory contains some extensions to prompt tuning beyond the experiments
found in our paper.

## Masking

The first extension is the ability to control visibility in attention via
masking. In the default setup, all input positions can attend within the input
and to all prompt positions. Similarly, all prompt tokens can attend to other
prompt positions as well as to all of the input tokens. The masking functions
found here allow you to hide some types of postitions (input vs prompt) from the
other type.

Attention masking needs to be augmented when using prompts so that the prompt
positions are correctly attended to. This would be simpler in a *virtual tokens*
implementation of prompt tuning (where prompts are represented by explicit
tokens that are added to the model vocabulary), but would lose the fine grain
control over attention visibility these functions provide.

As such, we provide several mask creation functions in
[masks.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/extended/masks.py)
that are mask aware. They are used in slightly different situations.

We use the following symbols to denote allowed visibility when creating an
attention mask.

*   `"input->input"` :: Input tokens are allowed to attend to other input
    tokens. This is the standard setup in transformers and is always included.
*   `"input->prompt"` :: The input tokens are allowed to attend to the prompt.
    We want the prompt to control/influence the representation of other tokens
    so this is always enabled. Results in section 5.3 of
    [Zhang et al. (2021)](https://arxiv.org/abs/2106.10715) where restricting
    attention from the inputs to the prompts caused a massive drop in
    performance support this default.
*   `"prompt->input"` :: Allow the prompts to attend to the inputs. This results
    in prompts that are contextualized based on the specific input example.
*   `"prompt->prompt"` :: Allow the prompts to attend to other prompts. This
    results in a prompt that makes large changes in value from the first layer
    of the model to the last, but the changes are consistent across various
    examples.
*   `"prompt->prompt;causal"` :: Allow prompts to only attend to prompts that
    are in an earlier timestep as them. This is required for decoder only models
    and it may help turn the prompts into more of a sequence instead of a bag.

If none of the above `"prompt->X"` attentions are enabled the prompt can only
attend to itself meaning that the prompt value only changes based on the result
of the feed forward layers in the transformer.

We have found that the choice of prompt attention does not matter too much and
mostly just follow what is natural for the model we are prompting. That is,
`{"input->input", "input->prompt", "prompt->prompt", "prompt->input"}` for
encoder-decoders and `{"input->input", "input->prompt",
"prompt->prompt;causal"}` for decoder-only models.

*Note:* These functions assume the prompt is added to the front of the input, in
some cases it will work if the prompt is added to the end but one should double
check their prompts are correctly covered if they are putting them in
non-standard places.

### Encoder Masking

Prompt encoder masking is fairly straight forward, as it is typically fully
visible, this means we just need to have extra positions in the mask for the
prompts. This is handled in the `masks.prompt_encoder_attention_mask`. It also
supports specific attention strategies like having the prompt not be able to see
the input.

*Note:* When using full visibility, prompts can see everything and inputs can
see everything) this method should work for other prompt placement strategies.

### Decoder Masking

There are two different decoder settings when doing prompt tuning. The first is
when the decoder is part of an encoder-decoder and prompts have **not** been
applied to the decoder, only the encoder. The second option is when the prompts
are applied to the decoder, be it part of an encoder-decoder or a decoder-only
language model.

#### EncoderDecoder

In the case of encoder-decoder where there are no prompts on the decoder, all
that is required is that the `encoder_decoder_mask` includes extra visible
positions for the prompts. We do this via the `masks.add_fake_prompt` function
which just adds fake tokens for these positions.

*Note:* The decoder always has full visibility to the encoder inputs, therefore,
we can actually use this same function regardless of where the encoder prompts
are added, we will always just see everything.

#### Decoder Only

When the prompt is applied to the decoder, the decoder mask needs to be updated.
This can be done with the `masks.prompt_decoder_attention_mask`. This supports
creating both a fully causal mask, by passing `None` for
`decoder_causal_attention` and a prefix mask by passing a binary mask for the
`decoder_causal_attention` where a value of `1` denotes that a position is part
of the input.

*Note:* This method can be used to create a decoder mask regardless of prompt
position in the following cases. 1. You are using a fully causal mask and you
set allowable attentions to `{"prompts->prompts;causal"}` 2. You are using a
prefix mask and set the allowable attentions to `{"prompts->prompts",
"prompts->inputs"}`

## MultiTasking Prompts

*Note:* In multi-task prompting, various shape annotations can be subscripted
with `s` or `t` for the shared prompt or the task specific prompt specifically.
For example, `P_s` is the length of the shared prompt while `P_t` is the length
of the task specific prompt.

We also support MultiTask prompts where a prompt is broken into two parts. There
is a `shared_prompt` that is used for every task, and then a task specific
prompt. The task specific prompt is different for each task you are using. The
task specific prompt is then combined with the shared prompt to create the
actual prompt for that example. We offer the following combination strategies:

*   `AddMultiTaskPrompt` :: The shared prompt and task specific prompts are the
    same size and summed together.
*   `ConcatMultiTaskPrompt` :: The shared prompt and task specific prompts have
    the same number of features and are concatenated together along the sequence
    dimension.
*   `ConcatFeaturesMultiTaskPrompt` :: The shared prompt and task specific
    prompts are the same length and concatenated in the feature dimension. The
    sum of features need to match the model embedding size.
*   `ProjectMultiTaskPrompt` :: The task specific prompt is actually a flattened
    kernel and bias for an affine transformation that is applied to each
    timestep in the shared prompt.

These implementations are also found in
[multitask_prompts.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/extended/multitask_prompts.py)
and the module to add them to a model is found in
[train/multitask_prompts.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/extended/train/multitask_prompts.py).

Input data should be formatted so that the first token is a `task_index`. This
is a single *integer* token representing which task this example belongs to.

## Wayward Prompts

A simple method that tries to tease meaning from the learned prompt is to find
the text tokens whose embeddings are the nearest neighbors to the learned prompt
tokens under some distance metric (often the cosine distance).
[Khashabi et al. (2021)](https://arxiv.org/pdf/2112.08348.pdf) showed that it
is possible to control what these nearest neighbor text tokens are while still
having a learned prompt that solves the task. This is done by regularizing the
learned prompt towards the embedded represention of some discrete prompt. The
end result is a prompt that solves the task and the nearest neighbors are the
tokens in the discrete prompt.

[train/wayward.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/extended/train/wayward.py;l=1)
implements this regularization. An embedded discrete prompt is provided and the
squared L2 distance between the learned prompt and this embedded discrete
prompt is included in the loss (scaled by the hyper-parameter gamma).

An example configuration of this setup, including how to re-use prompt
initialization code to create the embedded discrete prompt can be found at
[configs/extended/models/wayward_t5_1_1_base_prompt.gin](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/extended/models/wayward_t5_1_1_base_prompt.gin).
