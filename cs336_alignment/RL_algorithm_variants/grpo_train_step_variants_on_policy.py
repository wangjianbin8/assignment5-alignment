import math
from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer


from cs336_alignment.hw1.compute_rollout_rewards import compute_rollout_rewards
from cs336_alignment.RL_algorithm_variants.compute_group_normalized_rewards_maxrl import compute_group_normalized_rewards
from cs336_alignment.hw1.tokenize_prompt_and_output import tokenize_prompt_and_output

from cs336_alignment.hw1.get_response_log_probs import get_response_log_probs
from cs336_alignment.hw1.compute_policy_gradient_loss_on_policy import compute_policy_gradient_loss
from cs336_alignment.RL_algorithm_variants.aggregate_loss_across_microbatch_constant import aggregate_loss_across_microbatch




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
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    # sequence：弱化长度影响
    # constant：保留 token 总数带来的权重差异

    normalization_constant: int | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | float]
]:
    if importance_reweighting_method != "none":
        raise NotImplementedError(
            "This on-policy train step currently only "
            "supports importance_reweighting_method='none'."
        )

    original_batch_size = len(rollout_responses)
    device = next(model.parameters()).device

    raw_rewards, reward_metadata = (
        compute_rollout_rewards(
            reward_fn,
            rollout_responses,
            repeated_ground_truths,
        )
    )

    advantages, advantage_metadata = (
        compute_group_normalized_rewards(
            raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
    )

    # ---------------------------------------
    # Prune only AFTER full-group advantages
    # ---------------------------------------
    nonzero_mask = advantages != 0
    nonzero_indices = torch.nonzero(
        nonzero_mask,
        as_tuple=False,
    ).squeeze(-1)

    kept_prompts = [
        repeated_prompts[i]
        for i in nonzero_indices.tolist()
    ]

    kept_responses = [
        rollout_responses[i]
        for i in nonzero_indices.tolist()
    ]

    kept_advantages = advantages[nonzero_indices]

    optimizer.zero_grad()

    microbatch_size = math.ceil(
        original_batch_size
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

    num_kept = len(kept_responses)

    if num_kept == 0:
        zero_loss = torch.zeros(
            (),
            device=device,
        )

        metadata = {
            **reward_metadata,
            **advantage_metadata,
            "loss": 0.0,
            "grad_norm": 0.0,
            "token_entropy": 0.0,
        }

        return zero_loss, metadata
     # --------------------------------------------------
    # 5. 一个个 microbatch forward + backward
    # --------------------------------------------------
    for start in range(
        0,
        num_kept,
        microbatch_size,
    ):

        end = min(
            start + microbatch_size,
            num_kept,
        )

        current_microbatch_size = end - start


        tokenized = tokenize_prompt_and_output(
            kept_prompts[start:end],
            kept_responses[start:end],
            tokenizer,
        )

        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        advantages_micro = (
            kept_advantages[start:end].to(device)
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
                importance_reweighting_method=importance_reweighting_method,
            )
        )


        # ----------------------------------------------
        # 5c. sequence normalization
        # ----------------------------------------------

        microbatch_loss = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=per_token_loss,
                mask=response_mask,
                loss_normalization=loss_normalization,
                normalization_constant=normalization_constant,
            )
        )


        # entropy也用同样的sequence averaging做logging
        '''
        loss normalization 是算法的一部分；entropy normalization 是日志指标的定义。
        '''
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
        if loss_normalization == "sequence":
            scaled_loss = (
                microbatch_loss * current_microbatch_size / original_batch_size
            )
  
        elif loss_normalization == "constant":
            scaled_loss = microbatch_loss 

        else:
            raise NotImplementedError(
                f"Unsupported loss_normalization: "
                f"{loss_normalization}"
            )

        '''
        对于loss:
        因为被裁掉的 sequence 本来 advantage=0，对 loss 的贡献就是 0，所以裁剪只是省计算，并没有改变原来的训练目标。
        因此分母仍要用 original_batch_size，否则会把剩下样本的梯度人为放大      

        Loss：pruning 只省计算，必须保持原目标不变，所以 denominator 用原来的。
        Entropy：只是统计实际 forward 的样本，所以 denominator 用 num_kept。
        最直观地说就是：
        Loss 里的被 prune 样本是“已知贡献为 0”；entropy 里的被 prune 样本是“根本没计算，不知道是多少”。
        '''
        total_loss += scaled_loss.detach()

        # 实际经过 model forward 的 kept sequences 的平均 entropy
        total_entropy += (
            microbatch_entropy.detach()
            * current_microbatch_size
            / num_kept
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
        **reward_metadata,
        **advantage_metadata,
    }


    return total_loss, metadata