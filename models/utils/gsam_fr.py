"""
MIT License

Copyright (c) 2025 [Baofeng Liao]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

GSAM-FR (Gradient-Structure-Aware Minimization with Flatness Regularization)
An innovative optimizer combining gradient structure awareness with flatness regularization.
"""

import tensorflow as tf


class GSAMFR:
    """GSAM-FR (Gradient-Structure-Aware Minimization with Flatness Regularization) optimizer.
    
    This optimizer combines gradient structure awareness with flatness regularization
    to improve generalization in deep learning models.
    """
    
    def __init__(self, base_optimizer, alpha, rho, lambda_reg=0.1, eps=1e-12, adaptive=False):
        """Initialize the GSAM-FR optimizer.
        
        Args:
            base_optimizer: The base optimizer (e.g., Adam, SGD)
            alpha: Gradient decomposition coefficient
            rho: Perturbation radius for sharpness-aware minimization
            lambda_reg: Flatness regularization coefficient
            eps: Small constant for numerical stability
            adaptive: Whether to use adaptive perturbation scaling
        """
        self.alpha = alpha
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer
        self.e_ws = None  # Perturbation vectors
        self.old_gradients = None  # Previous gradients for decomposition

    def _grad_norm(self, gradients):
        """Compute the global norm of a list of gradients.
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Global norm as a scalar tensor
        """
        return tf.linalg.global_norm(gradients)


    def first_step(self, gradients, trainable_vars):
        """First step of GSAM-FR: apply perturbation to model parameters.
        
        This step adds a perturbation to the current parameters based on gradients,
        moving towards a sharper region of the loss landscape.
        
        Args:
            gradients: Current gradients of the model
            trainable_vars: List of trainable variables
        """
        self.e_ws = []  # Reset perturbation vectors
        self.old_gradients = [tf.identity(g) for g in gradients]  # Store gradients for decomposition

        # Compute gradient norm and perturbation multiplier
        grad_norm = self._grad_norm(gradients)
        ew_multiplier = self.rho / (grad_norm + self.eps)

        # Apply perturbation to each variable
        for grad, var in zip(gradients, trainable_vars):
            e_w = tf.math.multiply(grad, ew_multiplier)  # Base perturbation
            if self.adaptive:
                e_w = tf.math.multiply(e_w, tf.square(var))  # Adaptive scaling
            var.assign_add(e_w)  # Add perturbation to variable
            self.e_ws.append(e_w)  # Store perturbation for later reversal

    def _gradient_decompose(self, gradients):
        """Decompose gradients into structure-aware components.
        
        This method decomposes the current gradients by projecting them
        orthogonally to the previous gradients, enabling structure-aware optimization.
        
        Args:
            gradients: Current gradients after perturbation
            
        Returns:
            List of decomposed gradients
        """
        # Compute inner product between old and new gradients
        inner_prod = tf.reduce_sum([
            tf.reduce_sum(tf.multiply(old_g, new_g))
            for old_g, new_g in zip(self.old_gradients, gradients)
        ])

        # Compute gradient norms
        new_grad_norm = self._grad_norm(gradients)
        old_grad_norm = self._grad_norm(self.old_gradients)

        # Calculate cosine similarity between gradient directions
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.eps)

        # Decompose gradients: new_grad - alpha * vertical_component
        decomposed_grads = []
        for old_g, new_g in zip(self.old_gradients, gradients):
            # Compute vertical component (orthogonal to new gradient direction)
            vertical = old_g - cosine * old_grad_norm * new_g / (new_grad_norm + self.eps)
            # Apply gradient decomposition
            decomposed_grads.append(new_g - self.alpha * vertical)

        return decomposed_grads

    def second_step(self, gradients, trainable_vars):
        """Second step of GSAM-FR: apply decomposed gradients with flatness regularization.
        
        This step computes the final gradients by combining structure-aware gradients
        with flatness regularization, then applies the update using the base optimizer.
        
        Args:
            gradients: Gradients computed at perturbed parameters
            trainable_vars: List of trainable variables
        """
        # Compute flatness regularization gradients
        flatness_grads = self._compute_efficient_flatness_grad(gradients)

        # Decompose gradients for structure-aware optimization
        gsam_grads = self._gradient_decompose(gradients)

        # Combine GSAM gradients with flatness regularization
        final_grads = [
            gsam_g + self.lambda_reg * flat_g
            for gsam_g, flat_g in zip(gsam_grads, flatness_grads)
        ]

        # Reverse the perturbation applied in first_step
        for var, e_w in zip(trainable_vars, self.e_ws):
            var.assign_sub(e_w)

        # Apply the final gradients using the base optimizer
        self.base_optimizer.apply_gradients(zip(final_grads, trainable_vars))

    def _compute_efficient_flatness_grad(self, gradients):
        """Compute efficient flatness regularization gradients.
        
        This method provides an efficient approximation of flatness regularization
        by using the current gradients directly when gradient norm is non-zero.
        
        Args:
            gradients: Current gradients
            
        Returns:
            List of flatness regularization gradients
        """
        grad_norm = self._grad_norm(gradients)
        if grad_norm == 0:
            # Return zero gradients if current gradients are zero
            return [tf.zeros_like(g) if g is not None else None for g in gradients]

        # Use current gradients as efficient flatness approximation
        return [tf.identity(g) if g is not None else None for g in gradients]
