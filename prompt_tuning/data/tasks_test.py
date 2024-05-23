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

"""Tests for tasks."""

from absl.testing import absltest
from absl.testing import parameterized
from prompt_tuning.data import features
from prompt_tuning.data import tasks  # pylint: disable=unused-import
import seqio


TEST_TASKS = []
for model_prefix in features.MODEL_TO_FEATURES:
  TEST_TASKS.extend([
      # Tests for SuperGLUE datasets.
      dict(
          testcase_name=f"{model_prefix}BoolQ",
          base=f"{model_prefix}super_glue_boolq_v102",
          taskless=f"{model_prefix}taskless_super_glue_boolq_v102",
          task_index=f"{model_prefix}task_index_super_glue_boolq_v102",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}CB",
          base=f"{model_prefix}super_glue_cb_v102",
          taskless=f"{model_prefix}taskless_super_glue_cb_v102",
          task_index=f"{model_prefix}task_index_super_glue_cb_v102",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}COPA",
          base=f"{model_prefix}super_glue_copa_v102",
          taskless=f"{model_prefix}taskless_super_glue_copa_v102",
          task_index=f"{model_prefix}task_index_super_glue_copa_v102",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}MultiRC",
          base=f"{model_prefix}super_glue_multirc_v102",
          taskless=f"{model_prefix}taskless_super_glue_multirc_v102",
          task_index=f"{model_prefix}task_index_super_glue_multirc_v102",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}ReCoRD",
          base=f"{model_prefix}super_glue_record_v102",
          taskless=f"{model_prefix}taskless_super_glue_record_v102",
          task_index=f"{model_prefix}task_index_super_glue_record_v102",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}RTE",
          base=f"{model_prefix}super_glue_rte_v102",
          taskless=f"{model_prefix}taskless_super_glue_rte_v102",
          task_index=f"{model_prefix}task_index_super_glue_rte_v102",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}WiC",
          base=f"{model_prefix}super_glue_wic_v102",
          taskless=f"{model_prefix}taskless_super_glue_wic_v102",
          task_index=f"{model_prefix}task_index_super_glue_wic_v102",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}WSC",
          base=f"{model_prefix}super_glue_wsc_v102_simple_eval",
          taskless=f"{model_prefix}taskless_super_glue_wsc_v102_simple_eval",
          task_index=f"{model_prefix}task_index_super_glue_wsc_v102_simple_eval",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}DPR",
          base=f"{model_prefix}dpr_v001_simple",
          taskless=f"{model_prefix}taskless_dpr_v001_simple",
          task_index=f"{model_prefix}task_index_dpr_v001_simple",
          split="test"),
      dict(
          testcase_name=f"{model_prefix}AX-b",
          base=f"{model_prefix}super_glue_axb_v102",
          taskless=f"{model_prefix}taskless_super_glue_axb_v102",
          task_index=f"{model_prefix}task_index_super_glue_axb_v102",
          split="test"),
      dict(
          testcase_name=f"{model_prefix}AX-g",
          base=f"{model_prefix}super_glue_axg_v102",
          taskless=f"{model_prefix}taskless_super_glue_axg_v102",
          task_index=f"{model_prefix}task_index_super_glue_axg_v102",
          split="test"),
      # Tests for GLUE datasets
      dict(
          testcase_name=f"{model_prefix}COLA",
          base=f"{model_prefix}glue_cola_v002",
          taskless=f"{model_prefix}taskless_glue_cola_v002",
          task_index=f"{model_prefix}task_index_glue_cola_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}SST2",
          base=f"{model_prefix}glue_sst2_v002",
          taskless=f"{model_prefix}taskless_glue_sst2_v002",
          task_index=f"{model_prefix}task_index_glue_sst2_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}MRPC",
          base=f"{model_prefix}glue_mrpc_v002",
          taskless=f"{model_prefix}taskless_glue_mrpc_v002",
          task_index=f"{model_prefix}task_index_glue_mrpc_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}QQP",
          base=f"{model_prefix}glue_qqp_v002",
          taskless=f"{model_prefix}taskless_glue_qqp_v002",
          task_index=f"{model_prefix}task_index_glue_qqp_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}STS-B",
          base=f"{model_prefix}glue_stsb_v002",
          taskless=f"{model_prefix}taskless_glue_stsb_v002",
          task_index=f"{model_prefix}task_index_glue_stsb_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}MNLI",
          base=f"{model_prefix}glue_mnli_v002",
          taskless=f"{model_prefix}taskless_glue_mnli_v002",
          task_index=f"{model_prefix}task_index_glue_mnli_v002",
          split="train"),
      dict(
          testcase_name=f"{model_prefix}MNLI-Mismatched",
          base=f"{model_prefix}glue_mnli_mismatched_v002",
          taskless=f"{model_prefix}taskless_glue_mnli_mismatched_v002",
          task_index=f"{model_prefix}task_index_glue_mnli_mismatched_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}MNLI-Matched",
          base=f"{model_prefix}glue_mnli_matched_v002",
          taskless=f"{model_prefix}taskless_glue_mnli_matched_v002",
          task_index=f"{model_prefix}task_index_glue_mnli_matched_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}QNLI",
          base=f"{model_prefix}glue_qnli_v002",
          taskless=f"{model_prefix}taskless_glue_qnli_v002",
          task_index=f"{model_prefix}task_index_glue_qnli_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}RTE-GLUE",
          base=f"{model_prefix}glue_rte_v002",
          taskless=f"{model_prefix}taskless_glue_rte_v002",
          task_index=f"{model_prefix}task_index_glue_rte_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}WNLI",
          base=f"{model_prefix}glue_wnli_v002",
          taskless=f"{model_prefix}taskless_glue_wnli_v002",
          task_index=f"{model_prefix}task_index_glue_wnli_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}AX",
          base=f"{model_prefix}glue_ax_v002",
          taskless=f"{model_prefix}taskless_glue_ax_v002",
          task_index=f"{model_prefix}task_index_glue_ax_v002",
          split="test"),
      # Test for GLUE Transfer Tasks
      dict(
          testcase_name=f"{model_prefix}QQP_from_MRPC",
          base=f"{model_prefix}glue_qqp_from_mrpc_v002",
          taskless=f"{model_prefix}taskless_glue_qqp_from_mrpc_v002",
          task_index=f"{model_prefix}task_index_glue_qqp_from_mrpc_v002",
          split="validation"),
      dict(
          testcase_name=f"{model_prefix}MRPC_from_qqp",
          base=f"{model_prefix}glue_mrpc_from_qqp_v002",
          taskless=f"{model_prefix}taskless_glue_mrpc_from_qqp_v002",
          task_index=f"{model_prefix}task_index_glue_mrpc_from_qqp_v002",
          split="validation"),
  ])


class TasksTest(parameterized.TestCase):

  @parameterized.named_parameters(TEST_TASKS)
  def test_task_types(self, base, taskless, task_index, split):
    base = seqio.get_mixture_or_task(base)
    taskless = seqio.get_mixture_or_task(taskless)
    task_index = seqio.get_mixture_or_task(task_index)

    base_ds = base.get_dataset(None, split, shuffle=False)
    taskless_ds = taskless.get_dataset(None, split, shuffle=False)
    task_index_ds = task_index.get_dataset(None, split, shuffle=False)

    base_example = next(iter(base_ds))
    taskless_example = next(iter(taskless_ds))
    task_index_example = next(iter(task_index_ds))

    base_input_text = (
        base_example["inputs_pretokenized"].numpy().decode("utf-8"))
    taskless_input_text = (
        taskless_example["inputs_pretokenized"].numpy().decode("utf-8"))
    task_index_input_text = (
        task_index_example["inputs_pretokenized"].numpy().decode("utf-8"))

    taskless_input = taskless_example["inputs"]
    task_index_input = task_index_example["inputs"]

    # Check that a single text token was removed for taskless and task_index.
    self.assertLen(base_input_text.split(),
                   len(taskless_input_text.split()) + 1)
    self.assertLen(base_input_text.split(),
                   len(task_index_input_text.split()) + 1)

    # Check that we have added a single token to the tokenized result for
    # task_index vs taskless.
    self.assertEqual(task_index_input.shape[0], taskless_input.shape[0] + 1)

    # We don't test anything else, we don't know how many tokens the dataset
    # label tends to split into so don't check the tokenized output vs the
    # other tasks.

  @parameterized.named_parameters(
      dict(
          testcase_name="mrpc_from_qqp",
          mixture="glue_mrpc_from_qqp_zeroshot_v002",
          train="glue_qqp_v002",
          validation="glue_mrpc_from_qqp_v002"),
      dict(
          testcase_name="qqp_from_mrpc",
          mixture="glue_qqp_from_mrpc_zeroshot_v002",
          train="glue_mrpc_v002",
          validation="glue_qqp_from_mrpc_v002"),
  )
  def test_zero_shot_transfer(self, mixture, train, validation):
    print(mixture)
    mixture = seqio.get_mixture_or_task(mixture)
    train_task = [t for t in mixture.tasks if t.name == train][0]
    # Read the training task's validation split because it will be quicker.
    train_ds = train_task.get_dataset(None, "validation", shuffle=False)
    train_targets = set(
        [b["targets_pretokenized"].numpy().decode("utf-8") for b in train_ds])
    validation_task = [t for t in mixture.tasks if t.name == validation][0]

    # Make sure the train targets are converted in the validation label space.
    # If they aren't mapped correctly we would get the default value of -1. So
    # make sure we don't get an -1 from postprocessing.
    for train_target in train_targets:
      postprocessed_train_target = validation_task.postprocess_fn(
          train_target, is_target=True)
      self.assertNotEqual(postprocessed_train_target, -1)


if __name__ == "__main__":
  absltest.main()
