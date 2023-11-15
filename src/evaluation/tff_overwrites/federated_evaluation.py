# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple implementation of federated evaluation."""

import collections
from typing import Optional, Callable, Union

import attr
import tensorflow as tf

import tensorflow_federated.python.learning.model as model_lib
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce


@attr.s(eq=False, frozen=True, slots=True)
class ModelWeights(object):
    """A container for the trainable and non-trainable variables of a `Model`.

    Note this does not include the model's local variables.

    It may also be used to hold other values that are parallel to these variables,
    e.g., tensors corresponding to variable values, or updates to model variables.
    """
    trainable = attr.ib()
    non_trainable = attr.ib()

    @classmethod
    def from_model(cls, model):
        py_typecheck.check_type(model, (model_lib.Model, tf.keras.Model))
        return cls(model.trainable_variables, model.non_trainable_variables)

    @classmethod
    def from_tff_result(cls, struct):
        py_typecheck.check_type(struct, structure.Struct)
        return cls(
            [value for _, value in structure.iter_elements(struct.trainable)],
            [value for _, value in structure.iter_elements(struct.non_trainable)])


def weights_type_from_model(
        model: Union[model_lib.Model, Callable[[], model_lib.Model]]
) -> computation_types.StructType:
    """Creates a `tff.Type` from a `tff.learning.Model` or callable that constructs a model.

    Args:
      model: a `tff.learning.Model` instance, or a no-arg callable that returns a
        model.

    Returns:
      A `tff.StructType` representing the TFF type of the `ModelWeights`
      structure for `model`.
    """
    if callable(model):
        # Wrap model construction in a graph to avoid polluting the global context
        # with variables created for this model.
        with tf.Graph().as_default():
            model = model()
    py_typecheck.check_type(model, model_lib.Model)
    return type_conversions.type_from_tensors(ModelWeights.from_model(model))


def build_federated_evaluation(
        model_fn,
        broadcast_process: Optional[measured_process.MeasuredProcess] = None,
        use_experimental_simulation_loop: bool = False):
    """Builds the TFF computation for federated evaluation of the given model.

    Args:
      model_fn: A no-arg function that returns a `tff.learning.Model`. This method
        must *not* capture TensorFlow tensors or variables and use them. The model
        must be constructed entirely from scratch on each invocation, returning
        the same pre-constructed model each call will result in an error.
      broadcast_process: a `tff.templates.MeasuredProcess` that broadcasts the
        model weights on the server to the clients. It must support the signature
        `(input_values@SERVER -> output_values@CLIENT)` and have empty state. If
        set to default None, the server model is broadcast to the clients using
        the default tff.federated_broadcast.
      use_experimental_simulation_loop: Controls the reduce loop function for
        input dataset. An experimental reduce loop is used for simulation.

    Returns:
      A federated computation (an instance of `tff.Computation`) that accepts
      model parameters and federated data, and returns the evaluation metrics
      as aggregated by `tff.learning.Model.federated_output_computation`.
    """
    # Construct the model first just to obtain the metadata and define all the
    # types needed to define the computations that follow.
    with tf.Graph().as_default():
        model = model_fn()
        model_weights_type = model_utils.weights_type_from_model(model)
        batch_type = computation_types.to_type(model.input_spec)

    @computations.tf_computation(model_weights_type,
                                 computation_types.SequenceType(batch_type))
    @tf.function
    def client_eval(incoming_model_weights, dataset):
        """Returns local outputs after evaluting `model_weights` on `dataset`."""
        with tf.init_scope():
            model = model_fn()
        model_weights = model_utils.ModelWeights.from_model(model)
        tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                              incoming_model_weights)

        def reduce_fn(prev_loss, batch):
            model_output = model.forward_pass(batch, training=False)
            return prev_loss + tf.cast(tf.reduce_mean(model_output.loss), tf.float64)

        dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
            use_experimental_simulation_loop)
        dataset_reduce_fn(
            reduce_fn=reduce_fn,
            dataset=dataset,
            initial_state_fn=lambda: tf.constant(0, dtype=tf.float64))
        return collections.OrderedDict(local_outputs=model.report_local_outputs())

    @computations.federated_computation(
        computation_types.FederatedType(model_weights_type, placements.SERVER),
        computation_types.FederatedType(
            computation_types.SequenceType(batch_type), placements.CLIENTS))
    def server_eval(server_model_weights, federated_dataset):
        if broadcast_process is not None:
            broadcast_output = broadcast_process.next(broadcast_process.initialize(),
                                                      server_model_weights)
            client_outputs = intrinsics.federated_map(
                client_eval, (broadcast_output.result, federated_dataset))
        else:
            client_outputs = intrinsics.federated_map(client_eval, [
                intrinsics.federated_broadcast(server_model_weights),
                federated_dataset
            ])
        return model.federated_output_computation(client_outputs.local_outputs)

    return server_eval
