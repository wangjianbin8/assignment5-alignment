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

    advantages = raw_rewards_or_advantages

    # (B,) -> (B, 1)
    # so the same sequence-level advantage
    # broadcasts across all response tokens.
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)

    # --------------------------------------------------
    # On-policy
    # --------------------------------------------------

    if importance_reweighting_method == "none":

        per_token_loss = (
            -advantages * policy_log_probs
        )

        return per_token_loss, {}

    # --------------------------------------------------
    # Off-policy methods require old policy log probs
    # --------------------------------------------------

    if old_log_probs is None:
        raise ValueError(
            "old_log_probs is required for "
            "off-policy importance reweighting."
        )

    # w_t = πθ / π0
    #     = exp(log πθ - log π0)
    importance_ratio = torch.exp(
        policy_log_probs - old_log_probs
    )

    # --------------------------------------------------
    # Unclipped token-level importance reweighting
    # --------------------------------------------------

    if importance_reweighting_method == "noclip":

        per_token_loss = (
            -advantages * importance_ratio
        )

        return per_token_loss, {}

    # --------------------------------------------------
    # PPO / GRPO clipping
    # --------------------------------------------------

    elif importance_reweighting_method == "grpo":

        if cliprange is None:
            raise ValueError(
                "cliprange is required when "
                "importance_reweighting_method='grpo'."
            )

        clipped_ratio = torch.clamp(
            importance_ratio,
            min=1.0 - cliprange,
            max=1.0 + cliprange,
        )

        unclipped_objective = (
            advantages * importance_ratio
        )

        clipped_objective = (
            advantages * clipped_ratio
        )

        per_token_loss = -torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        return per_token_loss, {}

    elif importance_reweighting_method == "gspo":
        raise NotImplementedError(
            "GSPO will be implemented later."
        )

    else:
        raise NotImplementedError(
            "Unsupported importance reweighting method: "
            f"{importance_reweighting_method}"
        )