# SPoT: Soft Prompt Transfer

Note: This is a work in progress.

Data and code for our paper
[SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://aclanthology.org/2022.acl-long.346)
, published at [ACL 2022](https://www.2022.aclweb.org/).

# Using SPoT

At its core SPoT is essentially transfer learning for prompts. A prompt learned
on some other task is used as the initialization point for training on a new
task.

1.  First we need to a pre-trained prompt that will be used for initialization.
    See the
    [Training A Prompt Section](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/README.md#training-a-prompt)
    of the main README on how to train a prompt using the Prompt Tuning code
    base. This prompt will be used for initialization and should generally be
    trained on some large pre-training mixture.
2.  Second we need the path to that prompt we just trained. Wherever T5X was
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

3.  Finally we train a new prompt on our target task, initializing the prompt
    from the file above. To do this, we train a prompt as above, this time on
    the actual downsteam task you care about, and use the gin config
[prompts/from_file.gin](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/configs/prompts/from_file.gin)
    to automatically set this up. Replace any other `--gin_file=prompts/*.gin`
    argument with `--gin_file=prompts/from_file.gin` (or place it after the
    `--gin_file=models/*.gin` argument if no prompt config was used). The
    inclusion of this gin file will require an additional gin cli override
    `--gin.PROMPT_FILE`. This override points to a numpy file that will be read
    and used as the initial prompt value, i.e.
    `--gin.PROMPT_FILE=${MODEL_DIR}/numpy_checkpoints/checkpoint_${step}/encoder.prompt.prompt.prompt`.

# How to Cite

If you make use of this code or idea please cite:

```bibtex
@inproceedings{vu-etal-2022-spot,
    title = "{SP}o{T}: Better Frozen Model Adaptation through Soft Prompt Transfer",
    author = "Vu, Tu  and
      Lester, Brian  and
      Constant, Noah  and
      Al-Rfou{'}, Rami  and
      Cer, Daniel",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.346",
    doi = "10.18653/v1/2022.acl-long.346",
    pages = "5039--5059",
}
```
