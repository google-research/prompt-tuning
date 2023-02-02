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

"""Utilities for testing."""

import jax.numpy as jnp


class ArrayAllCloseMatcher:
  """Allows arrays in mock asserts with jnp.allclose as the comparison."""

  def __init__(
      self, arr, rtol: float = 1e-5, atol=1e-08, equal_nan: bool = False):
    super().__init__()
    self.arr = arr
    self.rtol = rtol
    self.atol = atol
    self.equal_nan = equal_nan

  def __eq__(self, other):
    return jnp.allclose(
        self.arr,
        other,
        rtol=self.rtol,
        atol=self.atol,
        equal_nan=self.equal_nan)

  def __repr__(self):
    return repr(self.arr)


class ArrayEqualMatcher:
  """Allows arrays in mock asserts with jnp.array_equal as the comparison."""

  def __init__(self, arr, equal_nan: bool = False):
    self.arr = arr
    self.equal_nan = equal_nan

  def __eq__(self, other):
    return jnp.array_equal(
        self.arr, other, equal_nan=self.equal_nan)

  def __repr__(self):
    return repr(self.arr)
