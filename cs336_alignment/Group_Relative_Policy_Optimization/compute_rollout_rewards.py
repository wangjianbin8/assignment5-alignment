from collections.abc import Callable
import torch

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    format_rewards = []
    answer_rewards = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, ground_truth)
        rewards.append(reward_info["reward"])
        format_rewards.append(reward_info["format_reward"])
        answer_rewards.append(reward_info["answer_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)

    metadata = {
        "mean_reward":
            sum(rewards) / len(rewards),

        "mean_format_reward":
            sum(format_rewards) / len(format_rewards),

        "mean_answer_reward":
            sum(answer_rewards) / len(answer_rewards),
    }

    return raw_rewards, metadata
