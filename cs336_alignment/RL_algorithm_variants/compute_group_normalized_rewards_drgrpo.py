from typing import Literal

import torch


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal[
        "std", "none", "mean"
    ] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:

    # (rollout_batch_size,)
    #
    # ->
    #
    # (num_prompts, group_size)
    grouped_rewards = raw_rewards.reshape(
        -1,
        group_size,
    )

    group_means = grouped_rewards.mean(
        dim=1,
        keepdim=True,
    )

    group_stds = grouped_rewards.std(
        dim=1,
        keepdim=True,
    )

    # -----------------------------------
    # Baseline
    # -----------------------------------

    if baseline == "mean":

        advantages = (
            grouped_rewards - group_means
        )

    elif baseline == "none":

        advantages = grouped_rewards

    else:

        raise NotImplementedError(
            f"Unsupported baseline: {baseline}"
        )

    # -----------------------------------
    # Advantage normalization
    # -----------------------------------

    if advantage_normalizer == "std":

        advantages = advantages / (
            group_stds + advantage_eps
        )

    elif advantage_normalizer == "none":

        # Dr. GRPO:
        # do not divide by group std
        pass

    else:

        raise NotImplementedError(
            "Unsupported advantage normalizer: "
            f"{advantage_normalizer}"
        )

    # Restore flattened rollout ordering.
    advantages = advantages.reshape(-1)

    metadata = {
        "mean_reward":
            raw_rewards.mean().item(),

        "max_reward":
            raw_rewards.max().item(),

        "min_reward":
            raw_rewards.min().item(),
    }

    return advantages, metadata