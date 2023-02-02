# Copyright 2023 Google.
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

"""Tests for our test utils."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from prompt_tuning import test_utils

DTYPES = (jnp.bfloat16, jnp.float32, jnp.int32)


class ArrayMatcherTest(parameterized.TestCase):

  @parameterized.product(
      shape=[(2,), (4, 4), (1, 100)],
      dtype1=DTYPES,
      dtype2=DTYPES,
      numpy1=[True, False],
      numpy2=[True, False],
      matcher=[test_utils.ArrayAllCloseMatcher, test_utils.ArrayEqualMatcher])
  def test_equal(self, shape, dtype1, dtype2, numpy1, numpy2, matcher):
    x = jnp.ones(shape, dtype=dtype1)
    if numpy1:
      x = np.array(x)
    y = jnp.ones(shape, dtype=dtype2)
    if numpy2:
      y = np.array(y)
    self.assertEqual(matcher(x), y)

  @parameterized.product(
      shape=[(2,), (4, 4), (1, 100)],
      dtype1=DTYPES,
      dtype2=DTYPES,
      numpy1=[True, False],
      numpy2=[True, False],
      matcher=[test_utils.ArrayAllCloseMatcher, test_utils.ArrayEqualMatcher])
  def test_equal_in_mock(self, shape, dtype1, dtype2, numpy1, numpy2, matcher):
    x = jnp.ones(shape, dtype=dtype1)
    if numpy1:
      x = np.array(x)
    y = jnp.ones(shape, dtype=dtype2)
    if numpy2:
      y = np.array(y)
    self.assertEqual(matcher(x), y)
    self.assertIsNot(x, y)

    def run(fn):
      return fn(y)

    m = mock.MagicMock()
    run(m)
    m.assert_called_once_with(matcher(x))

  @parameterized.product(
      shape=[(2,), (4, 4), (1, 100)],
      dtype1=DTYPES,
      dtype2=DTYPES,
      numpy1=[True, False],
      numpy2=[True, False],
      matcher=[test_utils.ArrayAllCloseMatcher, test_utils.ArrayEqualMatcher])
  def test_not_equal(self, shape, dtype1, dtype2, numpy1, numpy2, matcher):
    x = jnp.ones(shape, dtype=dtype1)
    if numpy1:
      x = np.array(x)
    y = jnp.zeros(shape, dtype=dtype2)
    if numpy2:
      y = np.array(y)
    self.assertNotEqual(matcher(x), y)

  @parameterized.product(
      shape=[(2,), (4, 4), (1, 100)],
      dtype1=DTYPES,
      dtype2=DTYPES,
      numpy1=[True, False],
      numpy2=[True, False],
      matcher=[test_utils.ArrayAllCloseMatcher, test_utils.ArrayEqualMatcher],
  )
  def test_not_equal_in_mock(
      self, shape, dtype1, dtype2, numpy1, numpy2, matcher):
    x = jnp.ones(shape, dtype=dtype1)
    if numpy1:
      x = np.array(x)
    y = jnp.zeros(shape, dtype=dtype2)
    if numpy2:
      y = np.array(y)
    self.assertIsNot(x, y)

    def run(fn):
      return fn(y)

    m = mock.MagicMock()
    run(m)
    with self.assertRaises(AssertionError):
      m.assert_called_once_with(matcher(x))


if __name__ == "__main__":
  absltest.main()
