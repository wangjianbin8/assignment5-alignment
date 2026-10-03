import math
from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer


from cs336_alignment.hw1.compute_rollout_rewards import compute_rollout_rewards
from cs336_alignment.hw1.compute_group_normalized_rewards_grpo import compute_group_normalized_rewards
from cs336_alignment.hw1.tokenize_prompt_and_output import tokenize_prompt_and_output

from cs336_alignment.hw1.get_response_log_probs import get_response_log_probs
from cs336_alignment.hw1.compute_policy_gradient_loss_on_policy import compute_policy_gradient_loss
from cs336_alignment.hw1.aggregate_loss_across_microbatch_sequence import aggregate_loss_across_microbatch




def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,

    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal[
        "std", "none", "mean"
    ] = "std",

    # Importance reweighting
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,

    # Loss normalization
    loss_normalization: Literal[
        "sequence", "constant"
    ] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | float]
]:

    # --------------------------------------------------
    # 0. 这部分作业只支持 standard on-policy GRPO
    # --------------------------------------------------

    if baseline != "mean":
        raise NotImplementedError(
            f"Unsupported baseline: {baseline}"
        )

    if advantage_normalizer != "std":
        raise NotImplementedError(
            f"Unsupported advantage normalizer: "
            f"{advantage_normalizer}"
        )

    if importance_reweighting_method != "none":
        raise NotImplementedError(
            f"Unsupported importance reweighting method: "
            f"{importance_reweighting_method}"
        )

    if loss_normalization != "sequence":
        raise NotImplementedError(
            f"Unsupported loss normalization: "
            f"{loss_normalization}"
        )


    full_batch_size = len(rollout_responses)

    device = next(model.parameters()).device


    # --------------------------------------------------
    # 1. Reward
    # --------------------------------------------------

    raw_rewards, reward_metadata = (
        compute_rollout_rewards(
            reward_fn,
            rollout_responses,
            repeated_ground_truths,
        )
    )


    # --------------------------------------------------
    # 2. Group-normalized advantage
    # --------------------------------------------------

    advantages, advantage_metadata = (
        compute_group_normalized_rewards(
            raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
    )


    # --------------------------------------------------
    # 3. Tokenize prompt + rollout
    # --------------------------------------------------

    train_batch = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer,
    )


    # --------------------------------------------------
    # 4. 准备 gradient accumulation
    # --------------------------------------------------

    optimizer.zero_grad()

    microbatch_size = math.ceil(
        full_batch_size
        / gradient_accumulation_steps
    )

    total_loss = torch.zeros(
        (),
        device=device,
    )

    total_entropy = torch.zeros(
        (),
        device=device,
    )


    # --------------------------------------------------
    # 5. 一个个 microbatch forward + backward
    # --------------------------------------------------

    for start in range(
        0,
        full_batch_size,
        microbatch_size,
    ):

        end = min(
            start + microbatch_size,
            full_batch_size,
        )

        current_microbatch_size = end - start


        input_ids = train_batch[
            "input_ids"
        ][start:end].to(device)

        labels = train_batch[
            "labels"
        ][start:end].to(device)

        response_mask = train_batch[
            "response_mask"
        ][start:end].to(device)

        advantages_micro = (
            advantages[start:end].to(device)
        )


        # ----------------------------------------------
        # 5a. 当前 policy 的 token log-prob + entropy
        # ----------------------------------------------

        response_stats = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=True,
        )

        policy_log_probs = response_stats[
            "log_probs"
        ]

        token_entropy = response_stats[
            "token_entropy"
        ]


        # ----------------------------------------------
        # 5b. -A log pi
        # ----------------------------------------------

        per_token_loss, loss_metadata = (
            compute_policy_gradient_loss(
                raw_rewards_or_advantages=advantages_micro,
                policy_log_probs=policy_log_probs,
                importance_reweighting_method="none",
            )
        )


        # ----------------------------------------------
        # 5c. sequence normalization
        # ----------------------------------------------

        microbatch_loss = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=per_token_loss,
                mask=response_mask,
                loss_normalization="sequence",
            )
        )


        # entropy也用同样的sequence averaging做logging
        microbatch_entropy = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=token_entropy,
                mask=response_mask,
                loss_normalization="sequence",
            )
        )


        # ----------------------------------------------
        # 5d. 修正 gradient accumulation 的权重
        # ----------------------------------------------

        weight = (
            current_microbatch_size
            / full_batch_size
        )

        scaled_loss = (
            microbatch_loss * weight
        )


        # 保存logging用的整个batch loss
        total_loss = (
            total_loss
            + microbatch_loss.detach() * weight
        )

        total_entropy = (
            total_entropy
            + microbatch_entropy.detach() * weight
        )


        # ----------------------------------------------
        # 5e. 累积gradient
        # ----------------------------------------------

        scaled_loss.backward()


    # --------------------------------------------------
    # 6. Gradient norm + clipping
    # --------------------------------------------------

    if max_grad_norm is not None:

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

    else:

        grad_norms = [
            p.grad.detach().norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]

        if len(grad_norms) > 0:
            grad_norm = torch.stack(
                grad_norms
            ).norm(2)
        else:
            grad_norm = torch.tensor(
                0.0,
                device=device,
            )


    # --------------------------------------------------
    # 7. 只更新一次参数
    # --------------------------------------------------

    optimizer.step()

    optimizer.zero_grad()


    # --------------------------------------------------
    # 8. logging metadata
    # --------------------------------------------------

    metadata = {
        "loss": total_loss.item(),
        "grad_norm": grad_norm.detach(),
        "token_entropy": total_entropy.item(),

        # reward logger
        **reward_metadata,
    }


    return total_loss, metadata