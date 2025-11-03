# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
# Copyright (c) 2025 Baofeng Liao
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Implementation of our Transformer with Reversible Instance Normalization and Channel-Wise Attention."""
import logging
import tensorflow as tf
from tensorflow.keras import layers
import collections
import random
import numpy as np
from models.utils import RevNorm, SAM, GSAMFR, SpectralNormalizedAttention


class BaseModel(tf.keras.Model):
    """
    A base model class that integrates various enhancements including
    Reversible Instance Normalization and Channel-Wise Attention, and optionally,
    spectral normalization and SAM/GSAM optimization.

    Attributes:
        pred_len (int): The length of the output predictions.
        num_heads (int): The number of heads in the multi-head attention mechanism.
        d_model (int): The dimensionality of the embedding vectors.
        opt_strategy (int): Optimization technique selector:
                       - 0: Use Adam
                       - 1: Use SAM
                       - 2: GSAM-FR
        use_attention (bool): If True, enables the multi-head attention mechanism in the model.
        use_revin (bool): If True, applies Reversible Instance Normalization (RevIN) to the model.
        trainable (bool): Specifies if the model's weights should be updated or frozen during training.
        rho (float): The neighborhood size parameter for SAM/GSAM-FR optimization.
        alpha (float): The alpha parameter for GSAM gradient decomposition.
        spec (bool): If True, applies spectral normalization to the attention mechanism.
        seed (int): Random seed for reproducibility. If provided, sets random seeds for TensorFlow,
                   NumPy, and Python's random module.
    """

    def __init__(self, pred_len, num_heads=1, d_model=16, opt_strategy=0,
                 use_attention=None, use_revin=None,
                 trainable=None, rho=0.5, alpha=0.20, lambda_reg=1.0, spec=None, seed=None):
        super(BaseModel, self).__init__()
        self.pred_len = pred_len
        self.num_heads = num_heads
        self.d_model = d_model
        self.opt_strategy = opt_strategy  # 0: Adam, 1: SAM, 2: GSAM-FR
        self.use_attention = use_attention
        self.use_revin = use_revin
        self.rho = rho if (opt_strategy in [1, 2] and trainable) else 0.0
        self.alpha = alpha if opt_strategy == 2 else 0.0
        self.lambda_reg = lambda_reg if opt_strategy == 2 else 0.0
        self.spec = spec
        self.seed = seed

        self.rev_norm = RevNorm(axis=-2)
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = layers.Dense(pred_len)
        self.all_attention_weights = collections.deque(maxlen=2)
        self.all_dense_weights = collections.deque(maxlen=2)

        if self.spec:
            self.spec_layer = SpectralNormalizedAttention(num_heads=num_heads, key_dim=d_model)

        # Define trainability of attention layer
        self.attention_layer.trainable = trainable

    def call(self, inputs, training=False):
        """
        The forward pass for the model.

        Parameters:
            inputs (Tensor): Input tensor.
            training (bool): Whether the call is for training.

        Returns:
            Tensor: The output of the model.
        """
        x = inputs
        if self.use_revin:
            x = self.rev_norm(x, mode='norm')
        x = tf.transpose(x, perm=[0, 2, 1])

        if self.use_attention:
            attention_output = self._apply_attention(x)
            x = layers.Add()([x, attention_output])

        x = self.dense(x)
        outputs = tf.transpose(x, perm=[0, 2, 1])

        if self.use_revin:
            outputs = self.rev_norm(outputs, mode='denorm')

        return outputs

    def _apply_attention(self, x):
        """
        Applies the attention mechanism to the input tensor.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying attention.
        """
        if self.spec:
            attention_output, weights = self.spec_layer(x, x, return_attention_scores=True)
        else:
            attention_output, weights = self.attention_layer(x, x, return_attention_scores=True)

        self.all_attention_weights.append(weights.numpy())
        return attention_output

    def get_last_attention_weights(self):
        """Returns the attention weights from the last but one batch."""
        if len(self.all_attention_weights) > 1:
            return self.all_attention_weights[-2]
        return None

    def train_step(self, data):
        """
        Custom training step that supports Adam, SAM and GSAM optimization.
        """
        # Unpack the data
        x, y = data

        if self.opt_strategy == 0:
            # Standard Adam optimization
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        elif self.opt_strategy == 1:
            # SAM optimization
            sam_optimizer = SAM(self.optimizer, rho=self.rho, eps=1e-12)

            # First forward-backward pass
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                # y_expanded = tf.expand_dims(y, axis=1)
                # y_broadcasted = tf.broadcast_to(y_expanded, tf.shape(y_pred))
                # print(y_broadcasted)
                # lossmae = tf.reduce_mean(tf.abs(y_broadcasted - y_pred))
                # print(lossmae)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            gradients = tape.gradient(loss, self.trainable_variables)
            sam_optimizer.first_step(gradients, self.trainable_variables)

            # Second forward -backward pass
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            gradients = tape.gradient(loss, self.trainable_variables)
            sam_optimizer.second_step(gradients, self.trainable_variables)

        elif self.opt_strategy == 2:
            # GSAM optimization
            gsam_optimizer = GSAMFR(self.optimizer, rho=self.rho, alpha=self.alpha, lambda_reg=self.lambda_reg,eps=1e-12)

            # First forward-backward pass
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            gradients = tape.gradient(loss, self.trainable_variables)
            gsam_optimizer.first_step(gradients, self.trainable_variables)

            # Second forward-backward pass
            with tf.GradientTape(persistent=True) as tape:
                y_pred_perturbed = self(x, training=True)
                loss = self.compiled_loss(y, y_pred_perturbed, regularization_losses=self.losses)

            grads_adv = tape.gradient(loss, self.trainable_variables)
            gsam_optimizer.second_step(grads_adv, self.trainable_variables)
            # 用完记得释放
            del tape

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred_perturbed)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}