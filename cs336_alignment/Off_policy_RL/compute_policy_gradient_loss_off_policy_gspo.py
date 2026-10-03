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

    # --------------------------------------------------
    # Shape:
    #
    # advantages:
    #     (B,) -> (B, 1)
    #
    # policy_log_probs:
    #     (B, T)
    #
    # This allows the same sequence-level advantage
    # to broadcast across all response tokens.
    # --------------------------------------------------

    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)

    # ==================================================
    # 1. On-policy
    #
    # L_t = -A log pi_theta
    # ==================================================

    if importance_reweighting_method == "none":

        per_token_loss = (
            -advantages * policy_log_probs
        )

        return per_token_loss, {}

    # ==================================================
    # All off-policy methods require old policy probs.
    # ==================================================

    if old_log_probs is None:
        raise ValueError(
            "old_log_probs is required for "
            "off-policy importance reweighting."
        )

    # --------------------------------------------------
    # log rho_t
    #
    # rho_t
    #   = pi_theta / pi_0
    #
    # log rho_t
    #   = log pi_theta - log pi_0
    #
    # shape: (B, T)
    # --------------------------------------------------

    log_ratio = (
        policy_log_probs
        - old_log_probs
    )

    # ==================================================
    # 2. Unclipped token-level importance reweighting
    #
    # J_t = A * rho_t
    #
    # L_t = -A * rho_t
    # ==================================================

    if importance_reweighting_method == "noclip":

        importance_ratio = torch.exp(
            log_ratio
        )

        per_token_loss = (
            -advantages * importance_ratio
        )

        return per_token_loss, {}

    # ==================================================
    # 3. PPO / GRPO token-level clipping
    #
    # J_t =
    # min(
    #     A * rho_t,
    #     A * clip(rho_t, 1-eps, 1+eps)
    # )
    #
    # L_t = -J_t
    # ==================================================

    elif importance_reweighting_method == "grpo":

        if cliprange is None:
            raise ValueError(
                "cliprange is required when "
                "importance_reweighting_method='grpo'."
            )

        # rho_t = exp(log rho_t)
        importance_ratio = torch.exp(
            log_ratio
        )

        clipped_ratio = torch.clamp(
            importance_ratio,
            min=1.0 - cliprange,
            max=1.0 + cliprange,
        )

        unclipped_objective = (
            advantages
            * importance_ratio
        )

        clipped_objective = (
            advantages
            * clipped_ratio
        )

        objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        # PyTorch minimizes loss,
        # so negate the objective.
        per_token_loss = -objective

        return per_token_loss, {}

    # ==================================================
    # 4. GSPO
    #
    # s =
    # (
    #     product_t rho_t
    # )^(1 / L)
    #
    # Numerically stable:
    #
    # log s
    #   = 1/L * sum_t log rho_t
    #
    # s
    #   = exp(log s)
    #
    # J =
    # min(
    #     A * s,
    #     A * clip(s, 1-eps, 1+eps)
    # )
    # ==================================================

    elif importance_reweighting_method == "gspo":

        if cliprange is None:
            raise ValueError(
                "cliprange is required when "
                "importance_reweighting_method='gspo'."
            )

        if response_mask is None:
            raise ValueError(
                "response_mask is required when "
                "importance_reweighting_method='gspo'."
            )

        # --------------------------------------------------
        # response_mask:
        #   1 -> actual response token
        #   0 -> prompt/padding/non-response token
        #
        # shape: (B, T)
        # --------------------------------------------------

        mask = response_mask.to(
            dtype=log_ratio.dtype,
            device=log_ratio.device,
        )

        # --------------------------------------------------
        # Actual response length for each sequence.
        #
        # shape: (B, 1)
        # --------------------------------------------------

        response_lengths = mask.sum(
            dim=-1,
            keepdim=True,
        )

        if torch.any(response_lengths == 0):
            raise ValueError(
                "GSPO requires every sequence to contain "
                "at least one response token."
            )

        # --------------------------------------------------
        # log s
        #
        # = 1 / len(y)
        #   * sum_{t in response} log rho_t
        #
        # shape: (B, 1)
        # --------------------------------------------------

        sequence_log_ratio = (
            (log_ratio * mask).sum(
                dim=-1,
                keepdim=True,
            )
            / response_lengths
        )

        # --------------------------------------------------
        # s = exp(log s)
        #
        # This is the geometric-mean importance ratio.
        #
        # shape: (B, 1)
        # --------------------------------------------------

        sequence_ratio = torch.exp(
            sequence_log_ratio
        )

        # --------------------------------------------------
        # PPO-style clipping, but now on the
        # sequence-level GSPO ratio s.
        # --------------------------------------------------

        clipped_sequence_ratio = torch.clamp(
            sequence_ratio,
            min=1.0 - cliprange,
            max=1.0 + cliprange,
        )

        unclipped_objective = (
            advantages
            * sequence_ratio
        )

        clipped_objective = (
            advantages
            * clipped_sequence_ratio
        )

        sequence_objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        # PyTorch minimizes loss.
        sequence_loss = -sequence_objective

        # --------------------------------------------------
        # Existing downstream code expects a per-token
        # tensor of shape (B, T).
        #
        # Every token in the same response shares the same
        # GSPO sequence-level objective.
        #
        # With sequence loss aggregation later:
        #
        #   1/L * sum_t sequence_loss
        #
        # = sequence_loss
        #
        # --------------------------------------------------

        per_token_loss = sequence_loss.expand_as(
            policy_log_probs
        )

        return per_token_loss, {}

    # ==================================================
    # Unsupported method
    # ==================================================

    else:

        raise NotImplementedError(
            "Unsupported importance reweighting method: "
            f"{importance_reweighting_method}"
        )