import math
from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from cs336_alignment.Group_Relative_Policy_Optimization.compute_rollout_rewards import (
    compute_rollout_rewards,
)
from cs336_alignment.RL_algorithm_variants.compute_group_normalized_rewards_maxrl import (
    compute_group_normalized_rewards,
)
from cs336_alignment.Group_Relative_Policy_Optimization.tokenize_prompt_and_output import (
    tokenize_prompt_and_output,
)
from cs336_alignment.Group_Relative_Policy_Optimization.get_response_log_probs import get_response_log_probs
from cs336_alignment.Off_policy_RL.compute_policy_gradient_loss_off_policy_gspo import compute_policy_gradient_loss
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
    loss_normalization: Literal[
        "sequence", "constant"
    ] = "sequence",

    # sequence:
    #   weakens response-length effects
    #
    # constant:
    #   preserves token-count contribution
    normalization_constant: int | None = None,

) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | float],
]:

    # ==================================================
    # 0. Validate off-policy arguments
    # ==================================================

    if (
        importance_reweighting_method != "none"
        and old_log_probs is None
    ):
        raise ValueError(
            "old_log_probs is required for "
            "off-policy training."
        )

    if (
        importance_reweighting_method in {
            "grpo",
            "gspo",
        }
        and cliprange is None
    ):
        raise ValueError(
            "cliprange is required for "
            f"importance_reweighting_method="
            f"'{importance_reweighting_method}'."
        )

    original_batch_size = len(
        rollout_responses
    )

    if len(repeated_prompts) != original_batch_size:
        raise ValueError(
            "repeated_prompts and rollout_responses "
            "must have the same length."
        )

    if (
        len(repeated_ground_truths)
        != original_batch_size
    ):
        raise ValueError(
            "repeated_ground_truths and "
            "rollout_responses must have the same length."
        )

    if (
        old_log_probs is not None
        and old_log_probs.shape[0]
        != original_batch_size
    ):
        raise ValueError(
            "old_log_probs must have the same batch "
            "dimension as rollout_responses."
        )

    device = next(
        model.parameters()
    ).device

    # ==================================================
    # 1. Compute rewards on the FULL batch
    # ==================================================

    raw_rewards, reward_metadata = (
        compute_rollout_rewards(
            reward_fn,
            rollout_responses,
            repeated_ground_truths,
        )
    )

    # ==================================================
    # 2. Compute advantages on the FULL groups
    #
    # Important:
    # Do this BEFORE pruning.
    # ==================================================

    advantages, advantage_metadata = (
        compute_group_normalized_rewards(
            raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=(
                advantage_normalizer
            ),
        )
    )

    # ==================================================
    # 3. Prune zero-advantage sequences
    #
    # These sequences would have zero gradient anyway.
    # ==================================================

    nonzero_indices = torch.nonzero(
        advantages != 0,
        as_tuple=False,
    ).squeeze(-1)

    indices_list = (
        nonzero_indices.tolist()
    )

    kept_prompts = [
        repeated_prompts[i]
        for i in indices_list
    ]

    kept_responses = [
        rollout_responses[i]
        for i in indices_list
    ]

    kept_advantages = (
        advantages[nonzero_indices]
    )

    # --------------------------------------------------
    # Off-policy:
    # old_log_probs must be pruned with EXACTLY the
    # same sequence indices, otherwise rows no longer
    # correspond to the same responses.
    # --------------------------------------------------

    if old_log_probs is not None:

        old_indices = nonzero_indices.to(
            old_log_probs.device
        )

        kept_old_log_probs = (
            old_log_probs.index_select(
                dim=0,
                index=old_indices,
            )
        )

    else:

        kept_old_log_probs = None

    num_kept = len(
        kept_responses
    )

    # ==================================================
    # 4. Prepare optimization
    # ==================================================

    optimizer.zero_grad()

    # Keep the same microbatch size as before pruning.
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

    # ==================================================
    # Edge case:
    # all advantages == 0
    # ==================================================

    if num_kept == 0:

        zero_loss = torch.zeros(
            (),
            device=device,
        )

        metadata = {
            "loss": 0.0,
            "grad_norm": 0.0,
            "token_entropy": 0.0,
            **reward_metadata,
            **advantage_metadata,
        }

        return zero_loss, metadata

    # ==================================================
    # 5. Microbatch forward + backward
    # ==================================================

    for start in range(
        0,
        num_kept,
        microbatch_size,
    ):

        end = min(
            start + microbatch_size,
            num_kept,
        )

        current_microbatch_size = (
            end - start
        )

        # ----------------------------------------------
        # 5a. Tokenize the current kept microbatch
        # ----------------------------------------------

        tokenized = (
            tokenize_prompt_and_output(
                kept_prompts[start:end],
                kept_responses[start:end],
                tokenizer,
            )
        )

        input_ids = tokenized[
            "input_ids"
        ].to(device)

        labels = tokenized[
            "labels"
        ].to(device)

        response_mask = tokenized[
            "response_mask"
        ].to(device)

        advantages_micro = (
            kept_advantages[
                start:end
            ].to(device)
        )

        # ----------------------------------------------
        # 5b. Current policy:
        #
        # log pi_theta
        # ----------------------------------------------

        response_stats = (
            get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
        )

        policy_log_probs = (
            response_stats[
                "log_probs"
            ]
        )

        token_entropy = (
            response_stats[
                "token_entropy"
            ]
        )

        # ----------------------------------------------
        # 5c. Old policy:
        #
        # log pi_0
        #
        # old_log_probs were computed BEFORE this
        # inference batch started being trained.
        # ----------------------------------------------

        if (
            kept_old_log_probs
            is not None
        ):

            old_log_probs_micro = (
                kept_old_log_probs[
                    start:end
                ].to(device)
            )

            # ------------------------------------------
            # Important padding alignment:
            #
            # old_log_probs may have been computed on a
            # larger batch whose maximum padded length
            # was longer.
            #
            # The current microbatch may have a shorter
            # maximum length.
            #
            # With right-padding, the first T positions
            # correspond to the same actual tokens.
            # ------------------------------------------

            current_seq_len = (
                policy_log_probs.shape[1]
            )

            if (
                old_log_probs_micro.shape[1]
                < current_seq_len
            ):
                raise ValueError(
                    "old_log_probs has fewer token "
                    "positions than current "
                    "policy_log_probs."
                )

            old_log_probs_micro = (
                old_log_probs_micro[
                    :,
                    :current_seq_len,
                ]
            )

        else:

            old_log_probs_micro = None

        # ----------------------------------------------
        # Sanity check:
        #
        # log pi_theta and log pi_0 must refer to the
        # same token positions.
        # ----------------------------------------------

        if (
            old_log_probs_micro
            is not None
            and old_log_probs_micro.shape
            != policy_log_probs.shape
        ):
            raise ValueError(
                "old_log_probs and "
                "policy_log_probs must have "
                "the same shape after alignment. "
                f"Got old={old_log_probs_micro.shape}, "
                f"current={policy_log_probs.shape}."
            )

        # ----------------------------------------------
        # 5d. Policy-gradient loss
        #
        # none:
        #   -A log pi_theta
        #
        # noclip:
        #   -A rho_t
        #
        # grpo:
        #   token-level PPO/GRPO clipping
        #
        # gspo:
        #   sequence-level geometric-mean ratio
        # ----------------------------------------------

        per_token_loss, loss_metadata = (
            compute_policy_gradient_loss(
                raw_rewards_or_advantages=(
                    advantages_micro
                ),
                policy_log_probs=(
                    policy_log_probs
                ),
                importance_reweighting_method=(
                    importance_reweighting_method
                ),
                old_log_probs=(
                    old_log_probs_micro
                ),
                cliprange=cliprange,

                # GSPO needs this to compute:
                #
                # 1 / len(y)
                # * sum_t log rho_t
                response_mask=response_mask,
            )
        )

        # ----------------------------------------------
        # 5e. Aggregate policy-gradient loss
        # ----------------------------------------------

        microbatch_loss = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=(
                    per_token_loss
                ),
                mask=response_mask,
                loss_normalization=(
                    loss_normalization
                ),
                normalization_constant=(
                    normalization_constant
                ),
            )
        )

        # ----------------------------------------------
        # Entropy is a LOGGING metric.
        #
        # Keep its definition fixed as sequence-average
        # entropy regardless of which RL loss
        # normalization we use.
        # ----------------------------------------------

        microbatch_entropy = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=(
                    token_entropy
                ),
                mask=response_mask,
                loss_normalization="sequence",
            )
        )

        # ----------------------------------------------
        # 5f. Correct gradient-accumulation scaling
        # ----------------------------------------------

        if (
            loss_normalization
            == "sequence"
        ):

            # aggregate_loss returned:
            #
            #   1 / M
            #   sum_{i in microbatch} L_i
            #
            # But the original objective is:
            #
            #   1 / N_original
            #   sum_i L_i
            #
            # Therefore multiply by M/N_original.
            scaled_loss = (
                microbatch_loss
                * current_microbatch_size
                / original_batch_size
            )

        elif (
            loss_normalization
            == "constant"
        ):

            # Each microbatch already computes:
            #
            #   sum(losses_in_microbatch) / Z
            #
            # Summing backward passes produces:
            #
            #   total_sum / Z
            #
            # so no extra M/N scaling.
            scaled_loss = (
                microbatch_loss
            )

        else:

            raise NotImplementedError(
                "Unsupported "
                "loss_normalization: "
                f"{loss_normalization}"
            )

        # ----------------------------------------------
        # Logging:
        # loss uses the original objective.
        # ----------------------------------------------

        total_loss += (
            scaled_loss.detach()
        )

        # ----------------------------------------------
        # Entropy logging:
        #
        # Average ONLY over sequences that actually
        # went through model.forward().
        #
        # Pruned sequences have unknown entropy;
        # their true entropy is not zero.
        # ----------------------------------------------

        total_entropy += (
            microbatch_entropy.detach()
            * current_microbatch_size
            / num_kept
        )

        # ----------------------------------------------
        # 5g. Accumulate gradient
        # ----------------------------------------------

        scaled_loss.backward()

    # ==================================================
    # 6. Gradient norm + clipping
    # ==================================================

    if max_grad_norm is not None:

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
        )

    else:

        grad_norms = [
            p.grad.detach().norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]

        if len(grad_norms) > 0:

            grad_norm = (
                torch.stack(
                    grad_norms
                ).norm(2)
            )

        else:

            grad_norm = torch.tensor(
                0.0,
                device=device,
            )

    # ==================================================
    # 7. ONE optimizer update
    #
    # Important:
    # This function performs one train step.
    #
    # In 32x off-policy training, the outer training
    # loop calls this function 32 times for one rollout
    # batch, once for each train minibatch.
    # ==================================================

    optimizer.step()

    optimizer.zero_grad()

    # ==================================================
    # 8. Logging metadata
    # ==================================================

    metadata = {
        "loss": total_loss.item(),
        "grad_norm": grad_norm.detach(),
        "token_entropy": (
            total_entropy.item()
        ),
        **reward_metadata,
        **advantage_metadata,
    }

    return total_loss, metadata