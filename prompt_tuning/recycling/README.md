# Prompt Recycling

Note: This is a work in progress.

Data and code for our paper [Reducing Retraining by Recycling Parameter-Efficient Prompts](https://arxiv.org/abs/2208.05577).

# Usage

1.  First we need a source prompt that will be recycled. See the
    [Training A Prompt Section](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/README.md#training-a-prompt)
    of the main README on how to train a prompt using the Prompt Tuning code
    base. This prompt will be used as the input to the recycler.
2.  Second, we need the path to that prompt we just trained. Wherever T5X was
    configured to save models checkpoints (controlled by the `--model_dir`
    flag), there will be a directory called `numpy_checkpoints`. In it there are
    directories for each saved step (`checkpoint_${step}`) and within that is
    the prompt, saved as a numpy file. This file will have a name like
    `encoder.prompt.prompt.prompt` (for Encoder-Decoder models) which is the
    path to the parameter through the model PyTree, using `.` for scoping. We
    will need this file. So to recap the trained prompt will live at:

```shell
${MODEL_DIR}/numpy_checkpoints/checkpoint_${step}/encoder.prompt.prompt.prompt
```

3.  We need to train a recycler using the source and target models and then
    apply it to the source prompt. The [run\_recycle.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/recycling/run\_recycle.py) script is able to do
    this. It takes the commandline arguments `--source_model` and
    `--target_model` which should point to the T5X checkpoints of the source
    model (which you trained the prompt with) and the target model (which you
    want to use with the recycled prompt) respectivly. It also requires the path
    to the source prompt as the `--prompt_path` parameter. Set this to the value
    above. You can select which recycler to use with the `--recycler` parameter.
    Finally the `--output_path` paramaeter is needed to specify where to save
    the recycled prompt.
4.  Finally, it is time to run eval. Follow the instructions from the
    [Inference with a
    Prompt](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/README.md#inference-with-a-prompt)
    section, but set the `--gin.PROMPT_FILE` override to the `--output_path`
    used above.

## Large Scale Automatic Experiments

The [recycle.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/recycling/recycle.py) script can be used with one of the [config files](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/spot/recycling/configs/) to train recyclers and generate recycled prompts. It will produce a `.txt` file of CLI arguments that will be helpful in running all the recycling experiments.

# Recycler Implementations

Our recycler implementations include:

* `v2v-nn` :: The Jax based Neural network. Implemented in `JaxRecycler`.
* `v2v-lin` :: The linear projection learned via least squares. Implemented in `TFLstSqRecycler`.
* `lin-comb` :: The linear combination of target embeddings based on the source embeddings. Implemented in `LinearCombinationRecycler`.

# Vocabulary Filtering

The final list of our filtered vocabulary items can be found in [filtered-vocab-english-only.json](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/recycling/data/filtered-vocab-english-only.json)

# How to Cite

If you build on this code or these ideas please cite:

```bibtex
@article{lester-etal-2022-recycling
  title={{R}educing {R}etraining by {R}ecycling {P}arameter-{E}fficient {P}rompts},
  author={Lester, Brian and Yurtsever, Joshua and Shakeri, Siamak and Constant Noah},
  year={2022},
  journal={arXiv preprint arXiv:2208.05577},
  url={https://arxiv.org/abs/2208.05577},
}
```
