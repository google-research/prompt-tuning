# Pretrained Prompts

Prompts trained using T5 1.1 lm100k base as the frozen model.

Path                                           | Prompt Length | Batch Size | Topology | Partition    | Dataset | Split      | Metric   | Score     | Note
---------------------------------------------- | ------------: | ---------: | -------- | -----------: | ------- | ---------- | -------- | --------: |-----------------
pretrained_prompts/t5_1_1_lm100k_base/sst2.npy | 100           | 128        | v3-8     | (1, 1, 1, 1) | SST2    | validation | Accuracy | 95.07     | Class Label Init
pretrained_prompts/t5_1_1_lm100k_base/mrpc.npy | 100           | 32         | v3-8     | (1, 1, 1, 1) | MRPC    | validation | F1/Acc   | 89.7/85.3 | Class Label Init
pretrained_prompts/t5_1_1_lm100k_base/rte.npy  | 100           | 32         | v3-8     | (1, 1, 1, 1) | RTE     | validation | Accuracy | 68.6      | Class Label Init
