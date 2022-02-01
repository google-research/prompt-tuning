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

"""Tests for postprocessors."""

import unittest.mock as mock
from absl.testing import absltest
from prompt_tuning.data import postprocessors


class PostprocessorsTest(absltest.TestCase):

  def test_postprocess(self):
    gold = "gold"
    postproc_fn = mock.create_autospec(
        postprocessors.remove_extra_id_0, spec_set=True, return_value=gold)
    model_output = "output"
    pred_field = "my_preds"
    output_field = "model_output"
    example_fields = ("example1", "example2", "missing_example")
    example1 = "Some text"
    example2 = 42
    example = {"example1": example1, "example2": example2}

    target = postprocessors.postprocess_with_examples(
        postproc_fn,
        model_output,
        prediction_field=pred_field,
        prediction_text_field=output_field,
        example_fields=example_fields,
        extra_arg="Bye!",
        example=example)
    self.assertEqual(target[pred_field], gold)
    self.assertEqual(target[output_field], model_output)
    self.assertEqual(target[example_fields[0]], example1)
    self.assertEqual(target[example_fields[1]], example2)
    self.assertIsNone(target[example_fields[2]])
    postproc_fn.assert_called_once_with(
        model_output, extra_arg="Bye!", example=example)

  def test_postprocess_is_target(self):
    gold = 12
    postproc_fn = mock.create_autospec(
        postprocessors.remove_extra_id_0, spec_set=True, return_value=gold)
    model_output = "test"
    target = postprocessors.postprocess_with_examples(
        postproc_fn, model_output, is_target=True, extra_arg="Hi!")
    self.assertEqual(target, gold)
    postproc_fn.assert_called_once_with(
        model_output, is_target=True, extra_arg="Hi!")

  def test_zero_shot_positive(self):
    qqp = "duplicate"
    mrpc = "equivalent"
    self.assertEqual(
        qqp, postprocessors.mrpc_to_qqp(postprocessors.qqp_to_mrpc(qqp)))
    self.assertEqual(
        mrpc, postprocessors.qqp_to_mrpc(postprocessors.mrpc_to_qqp(qqp)))

  def test_zero_shot_negative(self):
    qqp = "not_duplicate"
    mrpc = "not_equivalent"
    self.assertEqual(
        qqp, postprocessors.mrpc_to_qqp(postprocessors.qqp_to_mrpc(qqp)))
    self.assertEqual(
        mrpc, postprocessors.qqp_to_mrpc(postprocessors.mrpc_to_qqp(qqp)))

  def test_postprocess_sequential(self):
    return_one = "one"
    return_two = 2
    return_three = "again!"
    func1 = mock.create_autospec(
        postprocessors.remove_extra_id_0,
        spec_set=True,
        return_value=return_one)
    func2 = mock.create_autospec(
        postprocessors.remove_extra_id_0,
        spec_set=True,
        return_value=return_two)
    func3 = mock.create_autospec(
        postprocessors.remove_extra_id_0,
        spec_set=True,
        return_value=return_three)

    seq = postprocessors.sequential(func1, func2, func3)

    extra_arg1 = "extra man!"
    input_ = 12

    result = seq(input_, "slip this in", extra_arg=extra_arg1)

    self.assertEqual(result, return_three)
    func1.assert_called_once_with(input_, "slip this in", extra_arg=extra_arg1)
    func2.assert_called_once_with(
        return_one, "slip this in", extra_arg=extra_arg1)
    func3.assert_called_once_with(
        return_two, "slip this in", extra_arg=extra_arg1)

  def test_postprocess_input_is_dict(self):
    example = {"prediction": "This is my prediction",
               "from_example": "Make sure this is in there",
               "not_grabbed": "We shouldn't need this"}
    postproc_example = {
        "dummy_prediction": " ".join(example["prediction"].split()[:-1]),
        "from_previous": "need me!"}
    result = postprocessors.postprocess_with_examples(
        lambda x, *a, **kw: x[::-1],
        postproc_example,
        example_fields=("from_example", "from_previous"),
        decoded_model_output_field="dummy_prediction",
        example=example,
    )
    gold = {
        "prediction": "ym si sihT",
        "prediction_pretokenized": "This is my",
        "from_example": example["from_example"],
        "from_previous": postproc_example["from_previous"]
    }
    self.assertEqual(result, gold)


if __name__ == "__main__":
  absltest.main()
