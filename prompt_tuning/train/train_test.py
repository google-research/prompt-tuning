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

"""Larger scale tests to make sure core assumptions aren't being broken.

This modules contains integration-style tests. It tests things that are critical
for prompt tuning, but easy to overlook, are being used. Some of these include:

 * Testing that only the prompt is being updated based on our default configs.
   In prompt tuning we only make changes to the prompt, the model is frozen,
   this test loads in our model, based on our prompt tuning configs, runs the
   loss function and parameter update, and then checks that only the prompt
   variables have changed.
 * Testing that model parameters are loaded from the checkpoint. The t5x partial
   loading mechanism will fill all parameters not in the checkpoint with ones
   that are initialized from scratch. This means that if there is a problem and
   the weight isn't loaded from the checkpoint training won't fail, it will just
   be running with random weights. This test checks that model parameters are
   actually loaded from the checkpoint.
 * Testing the DecoderOnly inference can run with the right shapes and the like.

"""

import os
import re
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import traverse_util
import gin
import jax
import jax.numpy as jnp
import numpy as np
# t5.data.tasks has transitive dependencies that are gin configurable (in things
# like the t5.data.glue_utils). We need to import these now, before we parse the
# gin configs. If we don't then these files will (most likely) get imported
# during the `import_modules` call that makes sure that our task is loaded. This
# would introduce new gin.configurable objects after parsing which isn't
# allowed.
import t5.data.tasks  # pylint: disable=unused-import
from t5x import checkpoints
from t5x import partitioning
from t5x import train as train_lib
from t5x import utils

# Work-around for GIN `runs` configs expecting to find
# the `train` at the root scope of the `__main__`.
train = train_lib.train

FLAGS = flags.FLAGS
TEST_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_data")


class PromptsTrainTest(parameterized.TestCase):


  @parameterized.named_parameters([
      dict(
          testcase_name="prompt_t5_1_1_tiny",
          config="prompt_tuning/configs/test/train_t5_1_1_tiny_prompt.gin",
          change_regex=".*/prompt/.*"),
      dict(
          testcase_name="multitask_t5_1_1_tiny",
          config="prompt_tuning/configs/extended/test/train_multi_task_t5_1_1_tiny_prompt.gin",
          change_regex=".*/(shared_prompt|task_prompts)/.*"),
  ])
  def test_only_prompts_are_updated(self, config, change_regex):
    gin.clear_config(clear_constants=True)
    # Create references to gin configurable versions of our model and the
    # partition config.
    configured_partitioner = gin.configurable(partitioning.PjitPartitioner)
    # Parse the gin file.
    gin.parse_config_files_and_bindings([config], "")
    model = gin.query_parameter("%MODEL").scoped_configurable_fn()
    # Our `configured_(model|partition_cfg)` has now been populated by gin, as
    # if its arguments were applied during the parsing. Now we can call it with
    # no arguments.
    partitioner = configured_partitioner()

    # Create spec for fake input data.
    input_shapes = {
        "encoder_input_tokens": (4, 512),
        "decoder_input_tokens": (4, 4),
        "decoder_target_tokens": (4, 4)
    }
    input_types = {
        "encoder_input_tokens": jnp.int32,
        "decoder_input_tokens": jnp.int32,
        "decoder_target_tokens": jnp.int32
    }

    # This t5x object manages the initialization of the optimizer and model
    # parameters.
    train_state_initializer = utils.TrainStateInitializer(
        init_fn=model.get_initial_variables,
        optimizer_def=model.optimizer_def,
        input_shapes=input_shapes,
        input_types=input_types,
        partitioner=partitioner)
    # Create the optimizer from scratch, we don't care about the weights just
    # how they change so we don't need to load from a checkpoint.
    train_state = train_state_initializer.from_scratch(
        init_rng=jax.random.PRNGKey(42),)
    train_state_axes = train_state_initializer.train_state_axes

    def update(train_state, batch):
      # Create a jax grad function based on our models loss.
      grad_fn = jax.value_and_grad(model.loss_fn, has_aux=True)

      # Run the forward and backward pass of the model.
      (loss, (weight_sum, metrics)), grad = grad_fn(train_state.params, batch,
                                                    jax.random.PRNGKey(0))
      del loss, weight_sum, metrics

      # Apply the gradients to get an optimizer with updated variables.
      return train_state.apply_gradient(grad, learning_rate=0.3)

    p_update = partitioner.partition(
        update,
        in_axis_resources=(train_state_axes,
                           partitioning.PartitionSpec("data")),
        out_axis_resources=(train_state_axes))

    # Create fake data to feed the model.
    batch = {
        "encoder_input_tokens": jnp.ones((4, 512)),
        "decoder_target_tokens": jnp.ones((4, 4)),
        "decoder_input_tokens": jnp.ones((4, 4)),
    }

    new_train_state = p_update(train_state, batch)

    # Flatten both optimizers so the parameters are /scopes/of/nested/params.
    # This makes comparing them easier.
    flat_train_state = traverse_util.flatten_dict(train_state.params.unfreeze())
    flat_train_state = {"/".join(k): v for k, v in flat_train_state.items()}

    flat_new = traverse_util.flatten_dict(new_train_state.params.unfreeze())
    flat_new = {"/".join(k): v for k, v in flat_new.items()}

    # Make sure that any variable that matches the change regex has been updated
    # and any variable that doesn't has not.
    for var, og_weight in flat_train_state.items():
      new_weight = flat_new[var]
      if re.fullmatch(change_regex, var):
        self.assertFalse(
            np.all(np.allclose(new_weight, og_weight)), f"'{var}' matches.")
      else:
        np.testing.assert_allclose(
            new_weight, og_weight, err_msg=f"non-prompt '{var}' mismatch.")

  @parameterized.named_parameters([
      dict(
          testcase_name="prompt_t5_1_1_tiny",
          config="prompt_tuning/configs/test/load_t5_1_1_tiny_prompt.gin",
          checkpoint="test_t5_1_1_tiny/checkpoint_3/checkpoint"),
      dict(
          testcase_name="multitask_t5_1_1_tiny",
          config="prompt_tuning/configs/extended/test/load_multi_task_t5_1_1_tiny_prompt.gin",
          checkpoint="test_t5_1_1_tiny/checkpoint_3/checkpoint"),
  ])
  def test_prompt_loading(self, config, checkpoint):
    gin.clear_config(clear_constants=True)
    configured_partitioner = gin.configurable(partitioning.PjitPartitioner)
    configured_checkpoint_cfg = gin.configurable(utils.CheckpointConfig)
    checkpoint = os.path.join(FLAGS.test_srcdir, TEST_DATA, checkpoint)
    gin.parse_config_files_and_bindings(
        [config], f"INITIAL_CHECKPOINT_PATH='{checkpoint}'")

    model = gin.query_parameter("%MODEL").scoped_configurable_fn()
    partitioner = configured_partitioner()
    checkpoint_cfg = configured_checkpoint_cfg()

    input_shapes = {
        "encoder_input_tokens": (4, 512),
        "decoder_input_tokens": (4, 512),
        "decoder_target_tokens": (4, 512)
    }
    input_types = {
        "encoder_input_tokens": jnp.int32,
        "decoder_input_tokens": jnp.int32,
        "decoder_target_tokens": jnp.int32
    }

    train_state_initializer = utils.TrainStateInitializer(
        init_fn=model.get_initial_variables,
        optimizer_def=model.optimizer_def,
        input_shapes=input_shapes,
        input_types=input_types,
        partitioner=partitioner)
    train_state = train_state_initializer.from_checkpoint(
        ckpt_cfgs=[checkpoint_cfg.restore],
        ds_iter=None,
        init_rng=jax.random.PRNGKey(0),
    )

    checkpoint_contents = checkpoints.load_t5x_checkpoint(checkpoint)

    flat_train_state = traverse_util.flatten_dict(train_state.params.unfreeze())
    flat_train_state = {"/".join(k): v for k, v in flat_train_state.items()}

    flat_checkpoint = traverse_util.flatten_dict(checkpoint_contents)
    flat_checkpoint = {"/".join(k): v for k, v in flat_checkpoint.items()}

    for opt_key, opt_value in flat_train_state.items():
      if opt_value is not None:
        if opt_key in flat_checkpoint:
          np.testing.assert_allclose(
              opt_value,
              flat_checkpoint[opt_key],
              err_msg=f"'{opt_key}' mismatch.")


if __name__ == "__main__":
  absltest.main()
