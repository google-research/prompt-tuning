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

"""Install prompt-tuning."""

import ast
import setuptools


def get_version(file_name: str, version_name: str = "__version__") -> str:
  """Find version by AST parsing to avoid needing to import this package."""
  with open(file_name) as f:
    tree = ast.parse(f.read())
    # Look for all assignment nodes in the AST, if the variable name is what
    # we assigned the version number too, grab the value (the version).
    for node in ast.walk(tree):
      if isinstance(node, ast.Assign):
        if node.targets[0].id == version_name:
          return node.value.s
  raise ValueError(f"Couldn't find assignment to variable {version_name} "
                   f"in file {file_name}")

with open("README.md") as fp:
  LONG_DESCRIPTION = fp.read()

_jax_version = "0.2.27"

setuptools.setup(
    name="prompt-tuning",
    version=get_version("prompt_tuning/__init__.py"),
    description="Prompt Tuning from Lester et al., 2021",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Google Inc.",
    author_email="no-reply@google.com",
    url="http://github.com/google-research/prompt-tuning",
    license="Apache 2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "": ["**/*.gin"],
    },
    scripts=[],
    install_requires=[
        "absl-py",
        "flax @ git+https://github.com/google/flax#egg=flax",
        "gin-config",
        f"jax>={_jax_version}",
        "numpy",
        "seqio-nightly",
        "t5",
        "tensorflow",
        "tensorflow_datasets",
        # Install from git as they have setup.pys but are not on PyPI.
        "t5x @ git+https://github.com/google-research/t5x@main#egg=t5x",
        "flaxformer @ git+https://github.com/google/flaxformer@main#egg=flaxformer",
    ],
    extras_require={
        "test": ["pytest>=6.0"],
        # TODO: mt5 and byt5 are not setup as python packages.
        # Figure out best way to bring them in as dependencies.
        "mt5": [],
        "byt5": [],
        "mrqa": ["pandas"],
        "tpu": [f"jax[tpu]>={_jax_version}"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "prompt tuning",
        "machine learning",
        "transformers",
        "neural networks",
        "pre-trained language models",
        "nlp",
        "jax",
        "flax",
        "t5",
        "t5x",
    ]
)
