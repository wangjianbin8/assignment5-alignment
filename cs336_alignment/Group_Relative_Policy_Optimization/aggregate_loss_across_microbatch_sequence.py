from typing import Literal

import torch


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal[
        "sequence", "constant"
    ] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:

    # 这道题目前只要求 standard GRPO:
    # 先对每条 sequence 的 response tokens 求平均，
    # 再对 batch 中所有 sequences 求平均。
    if loss_normalization != "sequence":
        raise NotImplementedError(
            f"Unsupported loss normalization: "
            f"{loss_normalization}"
        )

    # mask可能是bool/int，
    # 转成与loss相同dtype方便乘法和除法
    mask = mask.to(
        dtype=per_token_policy_gradient_loss.dtype
    )

    # 只保留 response token 的 loss
    masked_loss = (
        per_token_policy_gradient_loss * mask
    )

    # 每条 sequence 的 response token loss 总和
    # shape: (batch_size,)
    loss_sums = masked_loss.sum(dim=1)

    # 每条 sequence 有多少个 response token
    # shape: (batch_size,)
    response_lengths = mask.sum(dim=1)

    # 每条 sequence 内部平均
    # shape: (batch_size,)
    sequence_losses = (
        loss_sums / response_lengths
    )

    # 再对所有 sequences 求平均
    # scalar
    loss = sequence_losses.mean()

    return loss