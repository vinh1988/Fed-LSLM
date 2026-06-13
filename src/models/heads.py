"""Classification heads for Yes/No (closed) and Open-Ended questions."""

from __future__ import annotations

import torch
import torch.nn as nn


class YesNoHead(nn.Module):
    """
    Classification head for yes/no (closed-ended) questions.
    Architecture: Linear(input_dim, 256) -> ReLU -> Dropout -> Linear(256, 2)
    """

    def __init__(self, input_dim: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor of shape [N, input_dim]
        Returns:
            Logits of shape [N, 2]
        """
        return self.net(x)


class OpenEndedHead(nn.Module):
    """
    Classification head for open-ended questions.
    Architecture: Linear(input_dim, 512) -> ReLU -> Dropout -> Linear(512, vocab_size)
    """

    def __init__(self, input_dim: int, vocab_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor of shape [N, input_dim]
        Returns:
            Logits of shape [N, vocab_size]
        """
        return self.net(x)
