# Prompt Tuning Implementation Details

This document details how the code works, where it lives, and some of the goals
of the implementation.

*   [Modeling in T5X](#modeling-in-t5x)
*   [Creating Prompts](#creating-prompts)
*   [Adding Prompts to Models](#adding-prompts-to-models)
*   [Prompt-Only Updates](#prompt-only-updates)
*   [Gin, Factories, and Flaxformer](#gin-factories-and-flaxformer)
*   [Partial Loading](#partial-loading)
*   [Partial Saving](#partial-saving)
*   [Testing](#testing)

## Vocabulary and Notation

*   **B ::** The batch size.
*   **T ::** The sequence dimension of input, the number of tokens in an
    example.
*   **H ::** The hidden/embedding dimension of the model.
*   **P ::** The length of the prompt.
*   **Model Tuning ::** Transfer a model to a new task by updating all the
    parameters in the models. Commonly called fine-tuning.
*   **Prefix-LM ::** A language model that expects `inputs` and `targets`
    features in its batch dictionary. This kind of model can support a causal
    attention mask or an prefix attention mask.
*   **Causal Attention Mask ::** An attention mask where each timestep can only
    see the timesteps before it. See mask #2 in Figure 3 of the
    [T5 paper](https://arxiv.org/pdf/1910.10683.pdf).
*   **Prefix Attention Mask ::** An attention mask where there is some prefix of
    the input where attention has bidirectional visibility. See mask #3 in
    Figure 3 of the [T5 paper](https://arxiv.org/pdf/1910.10683.pdf). *Note:*
    The use of this mask implies the use a `Prefix-LM` but the use of a
    `Prefix-LM` doesn't imply the use of this mask.
*   **verbalizers ::** The string used to represent a class
    [(Schick and Schütze, 2021)](https://arxiv.org/pdf/2001.07676.pdf).

## Modeling in T5X

There are three levels of modeling when it comes to T5X.

1.  [T5X/models.py](https://github.com/google-research/t5x/tree/main/t5x/models.py) ::
    The outermost layer is the model classes in T5X. These are normal python
    classes (not subclasses of Flax's `nn.Module`) that include methods like
    `predict_batch` and `compute_logits`. They handle interacting with the
    underlying Flax module, making the Flax `init` and `apply` calls.
2.  [Flaxformer EncoderDecoder](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/t5/t5_architecture.py) ::
    This layer is a Flaxformer module that handles the execution flow of the
    underlying components. For example, the `EncoderDecoder` class which has
    `encode` and `decode` methods, references to the actual encoder and decoder
    model, and a `__call__` function that handles calling these in the correct
    order. This layer is also responsible for creating the attention masks which
    is the main reason we interact with it.
3.  [Flaxformer Encoder and Decoder](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/t5/t5_architecture.py) ::
    This layer is the Flaxformer modules that actually do the work, things like
    `Encoder`. We modify this layer to actually add the prompts to the input.

## Creating Prompts

Our approach to prompting uses a prompt module that generates the prompt
parameters which are added to the embedded input rather than using special
virtual tokens with updatable embeddings.

The core implementation of our prompts can be found in
[prompts.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/prompts.py).
The core Prompt module takes in the input tokens `[B, T]` and the embedded
inputs `[B, T, H]` and returns an un-batched prompt variable, `[P, H]`.

## Using Prompt in Training

[train/prompts.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/prompts.py)
includes modules that are used to help insert prompts during training. The
actual prompt implementations are designed to return just the prompts themselves
to allow for maximum flexibility in their use. These training modules take care
of actually combining the prompt with the embedded inputs.

## Adding Prompts To Models

All of our prompting layers are based on
[Flaxformer](https://github.com/google/flaxformer). They are generally small
subclasses that override a method in order to add a call to the prompt module.

There are also a few Flaxformer subclasses whose main responsibility is to
create updated attention masking that is aware of our prompt.

### Adding to Encoders

The `PromptEncoder` class in
[train/layers.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/layers.py)
is our subclass that adds a prompt to the embedded input. We see here the only
real changes are a new class attribute for creating the prompt module and the
call to the prompt module itself.

The `PromptEncoderDecoder` class in
[train/layers.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/layers.py)
is our subclass that updates masking. On the encoder side we use our new
`prompt_encoder_attention_mask` function to create a new encoder mask, this lets
us do things like fancy masking for the prompt, while our decoder only needs to
think there are `P` extra tokens in the input. The decoder always has full
visibility to the encoder tokens so we don't need any things fancy.

### Adding to Decoders

The `PromptDecoder` class in
[train/layers.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/layers.py)
adds prompts to the decoder itself. The `PromptDecoderOnly` class handles
updated masking.

*Note:* The `PromptDecoder` could be combined with the `PromptEncoder` to make a
`PromptEncoderPromptDecoder` that has prompts applied to both sides of the
model. It would need to use the masking techniques from the
`PromptEncoderDecoder` class to make the `encoder_decoder_mask` and from the
`PromptDecoderOnly` class for the `decoder_mask`. Similarly we could create a
class that is an encoder-decoder where the prompt is only applied to the
decoder. We haven't had a need to write these classes yet.

#### Decoding with Prompts

The majority of prompt tuning has been able to be applied under-the-hood, that
is, the user does not need to change the way they interact with the module,
outside of configuration, to use the prompt. However, when doing decoding of a
model where the prompt has been applied to the decoder itself changes need to be
made, things like the size of the autoregressive cache, the number of timesteps
to fill in a single shot, and more all change when the prompt need to be
factored in. The `PromptPrefixLanguageModel` class in
[train/models.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/models.py)
is our implementation of decoding with prompts added.

## Prompt-Only Updates

The goal of prompt tuning is for the prompt to be updatable while the original
model is frozen. Hence, we need a mechanism that only applies the gradients to
the prompt when updating the model weights. Our solution is based on the
[flax.optim.MultiOptimizer](https://flax.readthedocs.io/en/latest/flax.optim.html#flax.optim.MultiOptimizer).
This takes a sequence of tuples of traversal and optimizer objects. The
traversal object takes a parameter name (that has been flattened by replacing
nesting with `/` scoping) and applies a filter function to it. If the filter
returns `True` the associated optimizer is used to update that parameter. The
`param_state` of a parameter that is not touched by any optimizer will be
`None`.

Our traversal filter function is a sequence of regular expressions. If any of
the provided regexes match the parameter's name then that parameter will be
updated.

We have several utilities required to use this multi-optimizer and configure it
via gin.

*   `match_any` in
    [train/utils.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/utils.py)
    is used to create a filter function what will return true if the path
    argument matches any of the provided regexes.
*   Our `MultiOptimizer` fork in
    [train/optim.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/utils.py).
    In T5X parameter partitioning (deciding how to split parameters across
    multiple hosts in model-parallelism workloads) is done bottom up, where Flax
    modules emit a `params_axes` collection along with their parameters. These
    are used to assign logical names to parameter dimensions and those names to
    used to create partitioning rules. This setup currently requires us to use
    the version of
    [Adafactor](https://proceedings.mlr.press/v80/shazeer18a/shazeer18a.pdf)
    that is bundled with T5X instead of the one in Flax. This version of
    Adafactor has a slightly different API and the default one and this fork is
    designed to deal with that. The fork also has a different constructor
    signature to get around the fact that gin cannot bind to variadic parameters
    (parameters that can accept multiple input values, like `*args` in Python)
    by converting `*travsreals_and_optimizers` to a normal argument that expects
    a list and it then unpacks..


*Note:* Partial model training results in a checkpoint where the previous
optimizer states have been set to `None` and will be missing from the final
checkpoint. This means it is not possible to start another round of training
where a different set of variables are trained (with their optimizer states
getting reloaded) without a manual set of combining the original checkpoint
(with the optimizer states) and the new checkpoint (with the trained parameter).
As of `2022/01/07` learning rate schedules cannot be used to simulate partial
training because they cause changes to the optimizer states, even when the
learning rate is zero.

## Partial Loading

Flax and T5X generally assume that the optimizer state defined by the config
matches the optimizer state defined by the checkpoint. This is not the case for
prompt tuning because of the new `prompt` parameter.

T5X lets us use the `assignment_map` field of the
`utils.RestoreCheckpointConfig` to mark parameters as not expected to be in the
checkpoint. The assignment map is a sequence of tuples. The main use case is to
associate variable names that might differ between the model and on disk.
Normally if the first element is the variable name in the optimizer as defined
by the config and the second is the variable name in the optimizer as defined by
the checkpoint. If we set this second value to `None` then we mark this
parameter as skipped and we do not try to load it from disk. Coupled with the
`fallback_scratch` parameter which backfills any parameter not loaded from disk
with that is created via the modules normal initialization, we can load the main
model from the checkpoint while initializing the prompt from scratch.

For simplicity we tend to use a tuple like `((r".*prompt.*", None),)` to find
our prompt variable.

*Note:* The T5X `assignment_map` code uses `re.fullmatch` meaning the provided
regex need to match the string in its entirety, i.e. the `.*` before and after
`prompt` are required.

## Partial Saving

When training large models, for example T5 XXL, prompts can look like they are
rather large (at a length of 100 this is 409600 parameters), but they are still
a tiny fraction of the total number of parameters (only 0.0037% in this case).
As such, saving the whole model, which is mostly the same as the pre-trained
checkpoint is wasteful.

We use our `Checkpointer` subclass in
[train/utils](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/utils.py)
to help alleviate this problem somewhat. Before saving the whole model, our
subclass will save any parameter, whose name matches one of a list of regexes,
to a Numpy array. The resulting file will live in
`${MODEL_DIR}/numpy_checkpoints/checkpoint_{$step}` and the name will be the
flattened parameter scope, with `.` instead of `/`. We save these Numpy
checkpoints into a different directory so that it is not effected by the T5X
checkpoint retention configuration.

*Note:* Although, we technically only need these Numpy files for our checkpoint,
we still do the full T5X checkpoint saving right now. This is because when T5X
recovers from preemption, it checks for its own saved data format and loads that
checkpoint (with things like the step) to continue training. We would need to
override things in the main train function, it be able to recognize that our
Numpy directory can be used to recover, and then also override the checkpoint
loading to load the default checkpoint and then overwrite variables with values
loaded from Numpy. Doing this would cut down on the time spent saving models,
but isn't worth it, given the lack of configurability in parts of the training
code. Instead, we set the number of checkpoints to keep as 1 so that we can
recover from preemption but have access to the model state at all steps without
copies of the large model.

## Gin, Factories and Flaxformer

A simple way to think about how gin works is to imagine it as python's
`functools.partial`. A gin configurable is the function and when gin parses the
config file it applies arguments in that config file to that function. For
example is we have the following configurable:

```python
def my_func(x, b, c):
  pass
```

and the configuration file

```python
my_func:
  x = 2
  b = "hello"
  c = 34
```

We can imagine that gin will apply this configuration to our function and we can
then call it with no parameters, like so:

```python
# gin's configuration parsing applies arguments
my_func = functools.partial(my_func, x=2, b="hello", c=34)
# Then when we call it we don't need to include arguments
my_func()
```

Flaxformer makes heavy use of this idea in their model configuration. The
majority of class attributes are expected to be factories, that is, instead of
directly passing a `nn.Module` into a Flaxformer model you pass a function that
when called, with no arguments, returns the correct `nn.Module`. In gin they
setup this up by having the class be the factory and gin is used to apply all
the arguments. Then when the Flaxformer `.setup` method calls each factory
function which returns a instance of the `nn.Module` you specified.

We follow this pattern in prompt tuning; the majority of configuration for
anything Flaxformer related is either a call that has constructor arguments
applied or a closure over the actual function that will be returned via the
factory call.

## Testing

To run the tests, install the package with the `[test]` option and then run
`pytest` from the root of the cloned repository.

When possible we try to test all JAX based code under `jax.jit`. This can help
find bugs that would not be noticed in other situations.

Some methods, such as the prompt initializes need to have some of their
arguments marked as static when they are `jit`ed. These functions normally take
something like a shape as an argument. When used in Flax they work correctly but
applying jit directly to them results in the shape becoming a tracer object so
what should be accessible shape information becomes unusable tensor values.

When you are using a mock to test that some injectable module or function is
used correctly (for example, your test ends with
`.assert_called_once_with(...)`) you cannot `jit` the method. This results in
errors about how the tracer object has leaked. You can use a mock in a `jit`ed
method to control outputs however.

We use `mock.create_autospec` and `mock.path.object(obj, "attr", autospec=True)`
when possible. This validates that the mock is not being called with methods it
does not have. However, this can cause issues, especially when you autospec a
class and get an object back with `instance=True`. If you try to call
`.assert_called_once_with` you will get an error because the `self` parameter is
not handled correctly. In these cases you need to check if called are equal
using `self.assertEqual` and `mock.call`

We provide a test utility, `ArrayMatcher` to help make assertions about
function calls that take jax/numpy arrays are arguments.
`.assert_called_once_with` does not correctly handle array parameters that are
not the same object. The equality check it does first does an `is` check before
calling the (possibly much more expensive) `__eq__` check. This means if the
array you mock was called with is the **same** object as the one you are
checking in the assert, the test would pass, but if they are **different**
objects with the same value, it would fail. The `ArrayMatcher` class defines
an `__eq__` method that uses `np.testing.assert_allclose` to compare arrays.
By wrapping the arrays in the expected call (the arguments to
`.assert_called_once_with`) in this call we can use normal asserts with arrays
as arguments.

### Longer Running, More Integration-Style Tests

[train/train_test.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/train_test.py)
includes a few tests that are longer running and more of a large scale
integration test, making sure that the models, layers, configs, and prompts all
work together. One test is that when the model is updated, only the prompt
variable is changed. The second test is that model weights (except for the
prompt weight) is correctly loaded from the checkpoint.

