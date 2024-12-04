# Prompt Tuning

This is the code to reproduce the experiments from the EMNLP 2021 paper "The
Power of Scale for Parameter-Efficient Prompt Tuning"
[(Lester et al., 2021)](https://aclanthology.org/2021.emnlp-main.243/).

These models are built on [T5X](https://github.com/google-research/t5x), which
defines the model and training loop;
[Flaxformer](https://github.com/google/flaxformer), which defines the actual
model computation; [Flax](https://github.com/google/flax), which defines the low
level model layers; and [Jax](https://github.com/jax-ml/jax), which provides the
actual execution. Details of our implementation can be found
[here](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/implementation.md).

## Table of Contents

*   [Installation](#installation)
*   [Training a Prompt](#training-a-prompt)
    *   [Training a Prompt on a Pod Slice](#training-a-prompt-on-a-pod-slice)
    *   [Custom Dependencies](#custom-dependencies)
*   [Inference with a Prompt](#inference-with-a-prompt)
*   [Model Configuration](#model-configuration)
*   [Prompt Initialization](#prompt-initialization)
*   [Released Model Checkpoints](#released-model-checkpoints)
*   [Released Prompts](#released-prompts)
*   [Extra Resources](#extra-resources)
*   [ How to Cite](#how-to-cite)

## Installation

1.  Follow the first 3 steps in the
    [T5X installation instructions](https://github.com/google-research/t5x#installation)
    to create a cloud TPU VM. Also follow step 5 and create a Google Cloud
    Storage (GCS) bucket. We will read and write data to this bucket using a URI
    formatted like `gs://{bucket-name}/path/to/item/in/bucket`. This is where we
    will store cached datasets as well as model checkpoints and results. For
    ease of reference, some of the most common cloud commands for interacting
    with the TPU VMs are

```sh
# Create a Cloud TPU VM
$ gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
    --zone ${ZONE} \
    --accelerator-type v3-8 \
    --version v2-alpha

# SSH into a Cloud TPU VM
$ gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE}

# Delete a Cloud TPU VM
$ gcloud alpha compute tpus tpu-vm delete ${TPU_NAME} --zone ${ZONE}
```

2.  You should now be at the command-line of the TPU VM instance. Clone the
    Prompt Tuning repository.

```sh
git clone --branch=main https://github.com/google-research/prompt-tuning
cd prompt-tuning
```

3.  Install the Prompt Tuning library.

```sh
python3 -m pip install .[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

If you run into an error where pip tries to install earlier and earliler
versions of dependencies (TensorFlow for example) until it tries to install
version `0.0.0` and then fails try adding `--use-deprecated=legacy-resolver` to
the install command. This error is related to required versions betweens
dependencies and the behavior is often called backtracking. If you use the
flag, it is possible that incompatible versions of libraries may be installed
and you should look out for warnings about mismatches in the output of the
install command.

*Note:* If you plan to hack on the internals of prompt tuning and need an
editable install (so changes in the cloned code are used when you run training)
run `pip` with the `-e` flag and you may need to delete the `pyproject.toml`
file if you are getting errors during installation.

To run the tests, install the package with the `[test]` (`python3 -m pip install
.[test] ...`) option and then run `python3 -m pytest` from the root of the
cloned repository.

## Training a Prompt

Training a prompt is similar to
[fine-tuning a model with T5X](https://github.com/google-research/t5x/blob/main/README.md#fine-tuning);
the main difference is that we have our own set of Prompt Tuning configuration
files to use.

We provide a demo script (`prompt_tuning/scripts/sst2-demo.sh`) that has all the
required parts for training a prompt. You can use this as a starting point, or
set `MODEL_DIR` and `TFDS_DATA_DIR` environment variables with paths to your
Google Cloud Storage bucket to run this script directly.

```sh
./prompt-tuning/prompt_tuning/scripts/sst2-demo.sh
```

To help with iteration speed, we tend to specify a lot more options the command
line rather than bundling all of the configuration into a single gin file. A few
options of note:

*   `--gin_search_paths` :: a comma separated list of directories to use as path
    prefixes for gin files. We can use `prompt_tuning.scripts.find_module
    ${module}` to find the install location of libraries that bundle
    configurations with them.
*   `--gin_file` :: The gin file to load. We tend to use paths relative starting
    with the library they are installed with, i.e.
    `prompt_tuning/configs/models/t5_1_1_base_prompt.gin` over
    `models/t5_1_1_base_prompt.gin` to avoid any confusion. Using the flag
    multiple time can be used to specify multiple gin files that will get merged
    together. Any configurations options set in multiple files will use the
    value from the last file in the list.
*   `--gin.{PARAM}={VALUE}` :: This general override flag will set `PARAM` to
    `VALUE`. This can be used to easily set configuration options without
    requiring them to be actual command line arguments. For example.
    `--gin.utils.SaveCheckpointConfig.keep=20` will save the last 20
    checkpoints.

### Training a Prompt on a Pod Slice

As models get larger, xl and xxl for example, they do not fit on the 8 TPUs that
come with a single TPU VM. In these cases we will need a slice of a TPU pod
(more information about TPU architecture and available configurations can be
found [here](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)). The
main difference between training a prompt on a single TPU VM and on a Pod slice
is that we now have multiple TPU VMs and will run the same SPMD JAX each VM,
this page has more information on
[multi-host](https://jax.readthedocs.io/en/latest/multi_process.html) JAX
programs. [This guide](https://cloud.google.com/tpu/docs/jax-pods) gives a quick
introduction to running JAX programs on a TPU Pod slice, but we will hit main
points here.

1.  Create a TPU Pod slice.
    [This page](https://cloud.google.com/tpu/docs/types-zones#us) lists which
    accelerator types are available in which zones. This is the same as
    creating a TPU VM above, except that we are requesting 32 TPUs instead of
    8.

```sh
$ gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
    --zone ${ZONE} \
    --accelerator-type v3-32 \
    --version v2-alpha
```

2.  Install the Prompt Tuning library. Given that we now have 4 TPU VM, each one
    has 8 of out TPUs, we want to forgo ssh'ing directly into the VM, as we
    would need to do that for each host. Instead, the Google Cloud SSH command
    allows use to specify a command to run with the `--command=` flag and that
    it should be run on all our VMs (called workers) with `--worker=all`.

```sh
$ gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone ${ZONE} \
  --worker=all \
  --command="git clone --branch=main https://github.com/google-research/prompt-tuning && cd prompt-tuning && "
python3 -m pip install . -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

3.  Write the script to train your prompt. We included a demo script
    (`/prompt_tuning/scripts/sst2-xxl-demo.sh`) the trains an prompt to solve
    the [SST2](https://nlp.stanford.edu/sentiment/) dataset using T5 1.1 lm100k
    XXL. You can use this as a starting point or just fill in the paths to your
    Google Cloud Storage bucket to specify where you want to save your results
    (`MODEL_DIR`) and where to cache TFDS data (`TFDS_DATA_DIR`), or set them as
    environment variables.

4.  Copy your training script each worker. If this is your first time running
    `scp` you may get error, run the `ssh-add /.../.ssh/google_compute_engine`
    command from the error message and try again.

```sh
$ gcloud alpha compute tpus tpu-vm scp sst2-xxl-demo.sh ${TPU_NAME}: \
  --zone=${ZONE}
  --worker=all
```

5.  Execute your training script.

```sh
$ gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone ${ZONE} \
  --worker=all \
  --command="./sst2-xxl-demo.sh"
```

If one of the workers has an error during training, you will be left with
processes that are using the TPUs on the other workers. This will stop you from
restarting your job until those processes a terminated and release the TPU. The
following command should end all these processes. You may see the `kill` command
man page come back from the worker who had the initial error.

```sh
$ gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone ${ZONE} \
  --worker=all \
  --command="sudo lsof -t /dev/accel0 | xargs kill -9"
```

### Custom Dependencies

To train prompts using custom parts, like your own dataset, follow the
[T5X Instructions on Custom Components](https://github.com/google-research/t5x#custom-components)

If you package your code as a pip-installable python package, you won't be bound
to a single directory, and you can use `python3 -m
prompt_tuning.scripts.find_module {your_module}` to help set the
`gin_search_paths` so that gin configs bundled in your library are findable.
*Note:* If you do plan to bundle gin configs in an installable package, make
sure that the directories that contain the config files have an `__init__.py` as
gin requires files to be in a python package.

If parts of your custom components are gin configurable, they need to be
explicitly imported in your gin files; if they end up getting imported after the
gin files are parsed, they will cause an error. If none of your dependencies
contain gin configurables, you can avoid writing a gin file by passing
`--gin.MIXTURE_OR_TASK_MODULE="'path.to.your.module'`. This will automatically
import your module and is convenient for when all you are doing is swapping out
datasets.


## Inference with a Prompt

Our suggested way to do inference with a prompt is to load the original
checkpoint used to initialize the model, and the prompt from a file. As
explained in this section about
[partial loading](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/implementation.md#partial-loading)
T5X supports loading some model parameters while initializing others from
scratch. We use this in conjunction with the `from_array` prompt initializer to
reload the frozen parameters from the original checkpoint and the prompt file a
file. The
[`configs/runs/prompt_eval.gin`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/runs/prompt_eval.gin)
sets up this configuration for you; you just have to supply a `PROMPT_FILE`. If
your model was trained with any of the `prompts/` config files, you can remove
them from the arguments to the evaluation script.

The included `sst2-demo-eval.sh` script shows an example of doing evaluation
this way. All that is needed is to set `EVAL_DIR` and `TFDS_DATA_DIR`
environment variables to the paths to store the output of evaluation and the
tensorflow datasets cache respectivly.

In T5X, the evaluation script assumes that your dataset has labels and outputs
the final results from your dataset's metric functions. The inference script
does not require labels and instead outputs your model's prediction. We
include an analogous
[`prompt_infer.gin`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/runs/prompt_infer.gin)
file to use with the inference script.

If you want to do inference or evaluation with the t5x checkpoint that is
produced from a prompt tuning training run, you can use the `(eval|infer).gin`
config from T5X directly. You will need to update the
`utils.RestoreChekcpointConfig` though. You should set `path` to the new
checkpoint, `assignment_map=()` and `fallback_to_scratch=False`.

## Model Configuration

All model, training, evaluation, saving, restoring, etc. configuration is done
via gin. See
[the gin-config repository](https://github.com/google/gin-config#readme) for a
general introduction to gin and
[this primer](https://github.com/google-research/t5x/blob/main/gin-primer.md)

We follow the T5X configuration layout:

*   `runs/` :: contains configs for the actual training of model. This is where
    things like dataset and evaluation configuration go.
*   `architectures/` :: contains configs for how the model works. This is where
    things like encoder-decoder vs decoder-only and embedding sharing are
    configured.
*   `models/` :: contains configs that set model specific parameters like the
    number of layers or the size of the embedding table. It also configures
    things like the T5X model wrapper used.
  *   `models/decoding/` :: contains easy to use configs to swap out how the
      model generates text during inference, includes configs for beam search
      and nucleus sampling.
  *   `models/sizes/` :: contains the various settings to create models of
      different sizes, these are combined with the default versions to create a
      sized version, for example,`t5_1_1_prompt.gin` + `sizes/large.gin`
      creates a T5 1.1 Large model. Some common combinations already available
      as gin files with the right includes (`t5_1_1_large_prompt.gin` for our
      example above). _Note:_ These size files need to come __after__ the main
      model file.
*   `prompts/` :: Our extra directory contains configs that set the `PROMPT` gin
    variable, allowing for easy switching of the prompt initialization based
    which prompt file is added as a `--gin_file` argument (it needs to come
    after the `models/` gin file),

### Order of gin config files

When specifying `--gin_file` arguments in the command line, the order matters.
The general order in which the gin files must be specified is:

1.  `models/*.gin`
1.  `prompts/*.gin`
1.  `models/sizes/*.gin*`
1.  `models/decoding/*.gin`
1.  `runs/*.gin`


### Required Fields

T5X has some required fields like `MIXTURE_OR_TASK_NAME` or
`TASK_FEATURE_LENGTHS`. We add two more:

*   `PROMPT_LENGTH` :: The length of the prompt we are using, this is used in a
    few different places to we require it as a gin macro we can reference in
    multiple places and ensure the values are in sync.
*   `PROMPT` :: This is the configuration of the actual prompt module that will
    be used in the Flaxformer `PromptX` subclasses.

*Note:* Prompt Tuning does not currently support packing of examples. This means
that our max target length only need to be long enough to fit the target for
each example. This means our `targets` key in the `TASK_FEATURE_LENGTHS` mapping
can be much shorter, for example around 4 for many SuperGLUE
[(Wang et al., 2019)](https://arxiv.org/abs/1905.00537) tasks, compared to 62
which is what the P5X default is.

## Prompt Initialization

There are several options for the initialization of the prompt parameter. We
support the various methods in section 3.2 our
[paper](https://aclanthology.org/2021.emnlp-main.243.pdf), as well as
initialization from a file. The latter allows one to do things like train on
BoolQ starting from a prompt learned on MNLI.

All initializers follow the flax initializer API of being a parameterized
function that returns a closure over the initialization function. The actual
initialization function always has the signature of

```python
def initializer(rng: Array, shape: Sequence[int]) -> Array:
  ...
```

We provide each initialization scheme as a gin configuration file in the
`configs/prompts` directory. They can be used by including the gin file with the
`--gin_file=path/to/configs/prompts/scheme.gin`. This file needs to come
**after** the main model file, otherwise the default (random uniform) method
will overwrite the one you selected. Some of these initialization methods will
require you to set extra gin values either though an override flag of in one of
your gin files.

**Random Uniform**

A standard, random initialization similar to what people have used for embedding
initialization. This is the default and no gin file is required. The scale of
the random values can be adjusted by overridding
`prompt_init/linen.initializers.uniform.scale=N`.

**Sampled Vocab**

Sample a token embedding to use as initialization for each prompt position with
the `from_sample_of_embeddings` initializer. You can limit the sampling to the
first `n` embeddings with the
`prompt_init/prompts.from_samples_of_embeddings.population_size` parameter.

This can be used with
[`--gin_file=prompt_tuning/configs/prompts/from_sampled_vocab.gin`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/prompts/from_sampled_vocab.gin).
This method uses the embedding table extracted from the initial model
checkpoint. You can also provide your own embedding file with
[`--gin_file=prompt_tuning/configs/prompts/from_sampled_vocab_numpy.gin`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/prompts/from_sampled_vocab_numpy.gin).
This method requires that you provide a value for `EMBEDDING_FILE` that is a
numpy array of the model's embedding table. This can be extracted from a model
checkpoint using
[prompt_tuning.scripts.extract_variable](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/scripts/extract_variable.py).

**Class Label**

We support initializing prompt timesteps with the embedding of class labels
(a.k.a. *verbalizers*) via the `from_embedded_list` initializer. Users providing
a list of words (class labels) to use. Each words is tokenized by a provided
vocab; embedded with a provided vocab table; aggregated, if need be, across
sub-tokens; and used to initialize a prompt time-step. If the provided tokens
don't cover the full prompt length, the missing tokens are initialized using
the provided fall back initializer.

We can match the paper, where unfilled prompt tokens are filled by sampling from
the embedding table, by composing this initialization with the one above. It can
be used with
[`--gin_file=prompt_tuning/configs/prompts/from_class_labels.gin`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/prompts/from_class_labels.gin).
This requires setting `CLASS_LABELS`, which is a list of the words that you want
to embed as prompt initialization. You can also provide your own embedding file
(which is the same as above) with
[`--gin_file=prompt_tuning/configs/prompts/from_class_labels_numpy.gin`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/prompts/from_class_labels_numpy.gin).
This additionally requires setting `EMBEDDING_FILE`.

**From String**

We also support initializing a prompt with the embedding of some string, often
used to start from a discrete prompt or a task description. This uses the
`from_embedded_string` initializer. The string is tokenized by the provided
vocabulary, each token is looked up in the provided embedding table, and the
resulting embedded representation of the string is used as a prompt
initialization. If the provided tokens don't cover the full prompt length, the
missing tokens are initialized using the provided fall back initializer.

_Note:_ The vocabulary just converts the string into a sequence of ids, you
will need to ensure that the string matches the result of any text formatting
(spaces around punctuation, etc.) that your SeqIO task does.

**From File**

You can also load a prompt from a file with the `from_array` initializer to
enable transfer across tasks. This is done with
[`--gin_file=prompt_tuning/configs/prompts/from_file.gin`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/prompts/from_file.gin).
This requires setting `PROMPT_FILE` with a path to the Numpy file with the
prompt to load. Numpy versions of the prompt are emitted by default when training,
but the prompt can also be extracted with the script mentioned above.

## Released Model Checkpoints

We have released T5X native checkpoints of the T5 1.1 checkpoints that have had
100K steps of language model adaptation.

*   **t5_1_1_lm100k_small** (~77 million parameters):
    [gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_small/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_small/checkpoint_1100000)
*   **t5_1_1_lm100k_base** (~250 million parameters):
    [gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000)
*   **t5_1_1_lm100k_large** (~800 million parameters):
    [gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_large/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_large/checkpoint_1100000)
*   **t5_1_1_lm100k_xl** (~3 billion parameters):
    [gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl/checkpoint_1100000)
*   **t5_1_1_lm100k_xxl** (~11 billion parameters):
    [gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xxl/checkpoint_1100000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_lm100k_xxl/checkpoint_1100000)

These are converted from the public
[Mesh TensorFlow checkpoints](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#lm-adapted-t511lm100k).

## Released Prompts

We have released pretrained prompts on a variety of tasks, and plan to add to
them over time.

Prompts can be found in the
[`pretrained_prompts`](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/pretrained_prompts)
directory. From there each sub-directory groups prompts by the model they were
trained for. The easiest way to reference these prompts that are bundled with
the library is:

```sh
  --PROMPT_FILE=`python3 -m prompt_tuning.scripts.find_module prompt_tuning`/pretrained_prompts/{MODEL_SIZE}/{PROMPT}.npy
```

Due to the inherent randomness of parallel computation, there are a few settings
that need to match between training and evaluation to get the exact same
numbers. Each model sub-directory has a `README.md` the specifies what these
settings should be. The most important settings to match are batch size, TPU
topology, and model parallelism partitioning. The tables include the scores you
should expect to see if you use these prompts in `t5x.eval`


## Extra Resources

This is a collection of additional resources about Prompt Tuning.

*   Presentations:
    *   EMNLP:
        [Video](https://blester125.com/presentations/prompt-tuning-emnlp-2021.html)
        [Slides](https://blester125.com/static/presentations/slides/Prompt-Tuning-Presentation-EMNLP-2021.pdf)
        [Poster](https://blester125.com/static/presentations/posters/Prompt-Tuning-EMNLP-2021-Poster.pdf)


## How to Cite

If you use this work as a jumping off point, please cite

```bibtex
@inproceedings{lester-etal-2021-power,
    title = "The Power of Scale for Parameter-Efficient Prompt Tuning",
    author = "Lester, Brian  and
      Al-Rfou, Rami  and
      Constant, Noah",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.243",
    doi = "10.18653/v1/2021.emnlp-main.243",
    pages = "3045--3059",
}
```

## Note

This is not an officially supported Google product.
