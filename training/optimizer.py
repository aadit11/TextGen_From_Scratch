"""
Optimizer and learning rate scheduler implementation for transformer training.

This module implements a custom learning rate scheduler with warmup as described
in the "Attention Is All You Need" paper, along with utility functions for
setting up the optimizer. The scheduler uses a special learning rate schedule
that increases linearly during warmup and then decreases proportionally to the
inverse square root of the step number.
"""

import torch.optim as optim
import torch

class CustomScheduler:
    """
    Implements a custom learning rate scheduler with warmup for transformer training.

    This scheduler implements the learning rate schedule from the "Attention Is All You Need"
    paper, which uses a special form of learning rate scheduling that:
    1. Increases linearly during the warmup period
    2. Decreases proportionally to the inverse square root of the step number after warmup

    The learning rate is computed as:
        lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate to schedule
        d_model (int): The dimension of the model, used in learning rate calculation
        warmup_steps (int, optional): Number of warmup steps. Defaults to 4000
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """
        Perform a single optimization step, updating the learning rate and optimizer.

        This method:
        1. Increments the step counter
        2. Computes the new learning rate
        3. Updates the learning rate for all parameter groups
        4. Performs the optimizer step
        """
        self.step_num += 1
        lr = self.learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()

    def learning_rate(self):
        """
        Compute the learning rate for the current step.

        Returns:
            float: The learning rate for the current step
        """
        return (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)

def get_optimizer(model, lr, d_model, warmup_steps=4000):
    """
    Create and configure an optimizer with custom learning rate scheduler.

    This function sets up an Adam optimizer with the recommended hyperparameters
    from the "Attention Is All You Need" paper and pairs it with the custom
    learning rate scheduler.

    Args:
        model (nn.Module): The model to optimize
        lr (float): Initial learning rate (will be modified by scheduler)
        d_model (int): The dimension of the model
        warmup_steps (int, optional): Number of warmup steps. Defaults to 4000

    Returns:
        tuple: A tuple containing:
            - torch.optim.Adam: The configured optimizer
            - CustomScheduler: The learning rate scheduler
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CustomScheduler(optimizer, d_model, warmup_steps)
    return optimizer, scheduler
