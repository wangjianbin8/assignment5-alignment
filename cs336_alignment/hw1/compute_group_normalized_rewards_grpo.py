from typing import Literal
import torch


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    if baseline != "mean":
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    if advantage_normalizer != "std":
        raise NotImplementedError(
            f"Unsupported advantage normalizer: "
            f"{advantage_normalizer}"
        )

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=-1, keepdim=True)
    group_stds = grouped_rewards.std(dim=-1, keepdim=True)
    advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)

    advantages = advantages.reshape(-1)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }

    return advantages, metadata