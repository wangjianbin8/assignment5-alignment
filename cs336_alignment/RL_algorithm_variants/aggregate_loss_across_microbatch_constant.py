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

    mask = mask.to(
        dtype=per_token_policy_gradient_loss.dtype
    )

    masked_loss = (
        per_token_policy_gradient_loss * mask
    )

    # -----------------------------------
    # Standard GRPO
    # -----------------------------------

    if loss_normalization == "sequence":

        # Sum token losses within each response.
        sequence_loss_sums = (
            masked_loss.sum(dim=1)
        )

        # Actual response lengths.
        sequence_lengths = (
            mask.sum(dim=1)
        )

        # Mean within each sequence.
        sequence_losses = (
            sequence_loss_sums
            / sequence_lengths
        )

        # Mean across sequences.
        loss = sequence_losses.mean()

        return loss

    # -----------------------------------
    # Dr. GRPO
    # -----------------------------------

    elif loss_normalization == "constant":

        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required "
                "when loss_normalization='constant'"
            )

        loss = (
            masked_loss.sum()
            / normalization_constant
        )

        return loss

    else:

        raise NotImplementedError(
            "Unsupported loss normalization: "
            f"{loss_normalization}"
        )