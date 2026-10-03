from typing import Literal

import torch


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError(
            f"Unsupported importance reweighting method: "
            f"{importance_reweighting_method}"
        )

    if raw_rewards_or_advantages.ndim == 1:
        advantages = raw_rewards_or_advantages.unsqueeze(-1)
    else:
        advantages = raw_rewards_or_advantages

    per_token_policy_gradient_loss = (
        -advantages * policy_log_probs
    )

    metadata = {}

    return per_token_policy_gradient_loss, metadata