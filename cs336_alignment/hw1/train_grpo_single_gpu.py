import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.hw1.grpo_train_step_standard_on_policy import (
    grpo_train_step,
)


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> list[dict]:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def extract_gsm8k_answer(answer: str) -> str:
    """
    GSM8K answer format:

        reasoning...
        #### 72

    Return:
        "72"
    """
    return answer.split("####")[-1].strip()


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(
    template: str,
    question: str,
) -> str:
    """
    The course r1_zero prompt should contain {question}.
    """
    if "{question}" not in template:
        raise ValueError(
            "Prompt template does not contain '{question}'. "
            "Check the prompt file."
        )

    return template.replace(
        "{question}",
        question,
    )


def truncate_at_answer_stop(
    text: str,
) -> str:
    """
    Mimic vLLM:

        stop = ["</answer>"]
        include_stop_str_in_output = True

    HF generate does not use the vLLM stop configuration here,
    so truncate after decoding.
    """
    stop = "</answer>"

    position = text.find(stop)

    if position == -1:
        return text

    return text[
        : position + len(stop)
    ]


# ============================================================
# Hugging Face rollout generation
#
# Single-GPU replacement for the course's second-GPU vLLM
# server.
# ============================================================


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    generation_batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:

    model.eval()

    responses = []

    for start in range(
        0,
        len(prompts),
        generation_batch_size,
    ):
        prompt_batch = prompts[
            start:start + generation_batch_size
        ]

        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # Because we use left padding, all rows have the
        # same padded prompt width.
        prompt_width = encoded[
            "input_ids"
        ].shape[1]

        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        generated_only = generated[
            :,
            prompt_width:
        ]

        texts = tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )

        for text in texts:
            responses.append(
                truncate_at_answer_stop(text)
            )

    return responses


# ============================================================
# Reward statistics
# ============================================================


def compute_reward_statistics(
    responses: list[str],
    ground_truths: list[str],
) -> dict[str, float]:

    total_rewards = []
    format_rewards = []
    answer_rewards = []

    for response, ground_truth in zip(
        responses,
        ground_truths,
    ):
        result = r1_zero_reward_fn(
            response,
            ground_truth,
        )

        total_rewards.append(
            float(result["reward"])
        )

        format_rewards.append(
            float(result["format_reward"])
        )

        answer_rewards.append(
            float(result["answer_reward"])
        )

    n = len(responses)

    return {
        "reward": sum(total_rewards) / n,
        "format_reward": sum(format_rewards) / n,
        "answer_reward": sum(answer_rewards) / n,
    }


def average_response_length(
    responses: list[str],
    tokenizer,
) -> float:

    if len(responses) == 0:
        return 0.0

    lengths = []

    for response in responses:
        token_ids = tokenizer(
            response,
            add_special_tokens=False,
        ).input_ids

        lengths.append(
            len(token_ids)
        )

    return sum(lengths) / len(lengths)


# ============================================================
# Validation
# ============================================================


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    validation_data: list[dict],
    prompt_template: str,
    device: torch.device,
    generation_batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[dict[str, float], list[dict]]:

    prompts = []
    ground_truths = []

    for example in validation_data:

        prompts.append(
            render_prompt(
                prompt_template,
                example["question"],
            )
        )

        ground_truths.append(
            extract_gsm8k_answer(
                example["answer"]
            )
        )

    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        generation_batch_size=generation_batch_size,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    reward_stats = compute_reward_statistics(
        responses,
        ground_truths,
    )

    reward_stats[
        "avg_response_length"
    ] = average_response_length(
        responses,
        tokenizer,
    )

    examples = []

    for i in range(
        min(5, len(responses))
    ):
        examples.append(
            {
                "question":
                    validation_data[i]["question"],
                "ground_truth":
                    ground_truths[i],
                "response":
                    responses[i],
            }
        )

    return reward_stats, examples


# ============================================================
# Logging helpers
# ============================================================


def scalar_value(value):
    """
    Convert scalar tensors to Python numbers for JSON/W&B.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().float().cpu().item()

        return None

    if isinstance(value, (int, float)):
        return value

    return None


def write_jsonl(
    path: Path,
    record: dict,
) -> None:

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def save_rollout_examples(
    output_dir: Path,
    step: int,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    max_examples: int = 8,
) -> None:

    path = (
        output_dir
        / f"rollouts_step_{step:04d}.jsonl"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for i in range(
            min(max_examples, len(responses))
        ):
            record = {
                "step": step,
                "prompt": prompts[i],
                "response": responses[i],
                "ground_truth":
                    ground_truths[i],
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Build one rollout batch
# ============================================================


def build_rollout_batch(
    train_data: list[dict],
    prompt_template: str,
    group_size: int,
    rollout_batch_size: int,
    rng: random.Random,
) -> tuple[
    list[str],
    list[str],
]:

    if (
        rollout_batch_size
        % group_size
        != 0
    ):
        raise ValueError(
            "rollout_batch_size must be divisible "
            "by group_size."
        )

    n_prompts = (
        rollout_batch_size
        // group_size
    )

    sampled_examples = rng.sample(
        train_data,
        n_prompts,
    )

    repeated_prompts = []
    repeated_ground_truths = []

    for example in sampled_examples:

        prompt = render_prompt(
            prompt_template,
            example["question"],
        )

        ground_truth = (
            extract_gsm8k_answer(
                example["answer"]
            )
        )

        # IMPORTANT:
        # Group members must remain contiguous because
        # compute_group_normalized_rewards reshapes:
        #
        #   (rollout_batch_size,)
        #       ->
        #   (n_prompts, group_size)
        #
        repeated_prompts.extend(
            [prompt] * group_size
        )

        repeated_ground_truths.extend(
            [ground_truth] * group_size
        )

    return (
        repeated_prompts,
        repeated_ground_truths,
    )


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    # ------------------------
    # Model/data
    # ------------------------

    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-0425-1B",
    )

    parser.add_argument(
        "--train-file",
        default="data/gsm8k/train.jsonl",
    )

    parser.add_argument(
        "--val-file",
        default="data/gsm8k/test.jsonl",
    )

    parser.add_argument(
        "--prompt-file",
        default=(
            "cs336_alignment/"
            "prompts/r1_zero.prompt"
        ),
    )

    # ------------------------
    # Dataset sizes
    # ------------------------

    parser.add_argument(
        "--n-train-examples",
        type=int,
        default=6400,
    )

    # Single-GPU debug default.
    # Official experiment should use >= 1024.
    parser.add_argument(
        "--n-val-examples",
        type=int,
        default=128,
    )

    # ------------------------
    # Training
    # ------------------------

    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    # Single-GPU defaults.
    # Course configuration is 256 / 8 / 32.
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    # ------------------------
    # Sampling
    # ------------------------

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )

    # Course config: 512.
    # Smaller default for your single GPU.
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
    )

    # ------------------------
    # Logging
    # ------------------------

    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--log-rollouts-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-dir",
        default="runs/grpo_single_gpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--wandb-project",
        default=None,
    )

    args = parser.parse_args()


    # ========================================================
    # Basic checks
    # ========================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this script."
        )

    if (
        args.rollout_batch_size
        % args.group_size
        != 0
    ):
        raise ValueError(
            "rollout_batch_size must be divisible "
            "by group_size."
        )

    if args.group_size < 2:
        raise ValueError(
            "group_size should be >= 2 when using "
            "std-normalized GRPO advantages."
        )


    set_seed(args.seed)

    rng = random.Random(args.seed)

    device = torch.device("cuda:0")


    # ========================================================
    # Output directory
    # ========================================================

    output_dir = (
        Path(args.output_dir)
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir / "metrics.jsonl"
    )


    # ========================================================
    # Optional WandB
    # ========================================================

    wandb_run = None

    if args.wandb_project is not None:

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is not installed. "
                "Install it or omit --wandb-project."
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"grpo-seed-{args.seed}",
        )


    # ========================================================
    # Dataset
    # ========================================================

    train_data = load_jsonl(
        args.train_file
    )

    val_data = load_jsonl(
        args.val_file
    )

    rng.shuffle(train_data)

    train_data = train_data[
        : min(
            args.n_train_examples,
            len(train_data),
        )
    ]

    val_data = val_data[
        : min(
            args.n_val_examples,
            len(val_data),
        )
    ]

    prompt_template = (
        load_prompt_template(
            args.prompt_file
        )
    )


    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model
        )
    )

    # Causal-LM generation should left-pad.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:

        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither pad_token "
                "nor eos_token."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    # ========================================================
    # Model
    # ========================================================

    print("Loading model...")

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
    )

    # Save activation memory during training.
    model.gradient_checkpointing_enable()

    # Required for gradient checkpointing.
    model.config.use_cache = False


    # ========================================================
    # Optimizer
    #
    # This is the optimizer specified by the assignment.
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )


    print(
        f"Model: {args.model}"
    )

    print(
        f"Train examples: {len(train_data)}"
    )

    print(
        f"Val examples: {len(val_data)}"
    )

    print(
        f"Rollout batch size: "
        f"{args.rollout_batch_size}"
    )

    print(
        f"Group size: {args.group_size}"
    )

    print(
        "Prompts per rollout batch: "
        f"{args.rollout_batch_size // args.group_size}"
    )


    # ========================================================
    # RL loop
    # ========================================================

    for step in range(
        1,
        args.num_steps + 1,
    ):

        print(
            f"\n========== STEP {step} =========="
        )


        # ----------------------------------------------------
        # 1. Construct current rollout batch
        # ----------------------------------------------------

        (
            repeated_prompts,
            repeated_ground_truths,
        ) = build_rollout_batch(
            train_data=train_data,
            prompt_template=prompt_template,
            group_size=args.group_size,
            rollout_batch_size=(
                args.rollout_batch_size
            ),
            rng=rng,
        )


        # ----------------------------------------------------
        # 2. Generate on-policy rollouts
        #
        # There is no vLLM weight sync on this single-GPU
        # version: the same HF model is both the rollout
        # policy and training policy.
        # ----------------------------------------------------

        rollout_responses = (
            generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=repeated_prompts,
                device=device,
                generation_batch_size=(
                    args.generation_batch_size
                ),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )
        )


        # ----------------------------------------------------
        # 3. Log raw rollout quality before updating
        # ----------------------------------------------------

        train_reward_stats = (
            compute_reward_statistics(
                rollout_responses,
                repeated_ground_truths,
            )
        )

        train_response_length = (
            average_response_length(
                rollout_responses,
                tokenizer,
            )
        )


        # ----------------------------------------------------
        # 4. One standard on-policy GRPO optimizer step
        # ----------------------------------------------------

        model.train()

        loss, train_metadata = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,

            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),

            max_grad_norm=(
                args.max_grad_norm
            ),

            reward_fn=r1_zero_reward_fn,

            repeated_prompts=(
                repeated_prompts
            ),

            rollout_responses=(
                rollout_responses
            ),

            repeated_ground_truths=(
                repeated_ground_truths
            ),

            group_size=args.group_size,

            baseline="mean",
            advantage_eps=1e-6,
            advantage_normalizer="std",

            importance_reweighting_method=(
                "none"
            ),

            old_log_probs=None,
            cliprange=None,

            loss_normalization="sequence",
            normalization_constant=None,
        )


        # ----------------------------------------------------
        # 5. Build training log
        # ----------------------------------------------------

        record = {
            "step": step,

            "train/loss":
                scalar_value(loss),

            "train/reward":
                train_reward_stats[
                    "reward"
                ],

            "train/format_reward":
                train_reward_stats[
                    "format_reward"
                ],

            "train/answer_reward":
                train_reward_stats[
                    "answer_reward"
                ],

            "train/avg_response_length":
                train_response_length,
        }


        # Preserve all statistics returned by your
        # grpo_train_step, including grad norm and entropy.
        for key, value in (
            train_metadata.items()
        ):
            scalar = scalar_value(value)

            if scalar is not None:
                record[
                    f"train/{key}"
                ] = scalar


        # Useful GPU debugging metric.
        record[
            "gpu/allocated_gb"
        ] = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        record[
            "gpu/reserved_gb"
        ] = (
            torch.cuda.memory_reserved()
            / 1024**3
        )


        # ----------------------------------------------------
        # 6. Validation
        # ----------------------------------------------------

        if (
            step == 1
            or step % args.eval_every == 0
        ):

            print("Running validation...")

            val_stats, val_examples = evaluate(
                model=model,
                tokenizer=tokenizer,
                validation_data=val_data,
                prompt_template=prompt_template,
                device=device,
                generation_batch_size=(
                    args.generation_batch_size
                ),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

            record[
                "val/reward"
            ] = val_stats[
                "reward"
            ]

            record[
                "val/format_reward"
            ] = val_stats[
                "format_reward"
            ]

            record[
                "val/answer_reward"
            ] = val_stats[
                "answer_reward"
            ]

            record[
                "val/avg_response_length"
            ] = val_stats[
                "avg_response_length"
            ]

            print(
                "Validation reward:",
                val_stats["reward"],
            )

            print(
                "Validation format reward:",
                val_stats["format_reward"],
            )

            print(
                "Validation avg response length:",
                val_stats[
                    "avg_response_length"
                ],
            )


        # ----------------------------------------------------
        # 7. Periodically save qualitative rollouts
        # ----------------------------------------------------

        if (
            step == 1
            or
            step % args.log_rollouts_every
            == 0
        ):
            save_rollout_examples(
                output_dir=output_dir,
                step=step,
                prompts=repeated_prompts,
                responses=rollout_responses,
                ground_truths=(
                    repeated_ground_truths
                ),
            )

            print("\nExample rollout:")
            print(
                rollout_responses[0]
            )


        # ----------------------------------------------------
        # 8. Write logs
        # ----------------------------------------------------

        write_jsonl(
            metrics_path,
            record,
        )

        if wandb_run is not None:
            wandb_run.log(
                record,
                step=step,
            )


        print(
            "Train reward:",
            train_reward_stats["reward"],
        )

        print(
            "Train format reward:",
            train_reward_stats[
                "format_reward"
            ],
        )

        print(
            "Loss:",
            scalar_value(loss),
        )

        if "train/grad_norm" in record:
            print(
                "Gradient norm:",
                record["train/grad_norm"],
            )

        if "train/token_entropy" in record:
            print(
                "Token entropy:",
                record[
                    "train/token_entropy"
                ],
            )


        # Can help reduce fragmentation on a small GPU.
        torch.cuda.empty_cache()


    # ========================================================
    # Finish
    # ========================================================

    final_dir = (
        output_dir / "final_model"
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        final_dir
    )

    tokenizer.save_pretrained(
        final_dir
    )

    if wandb_run is not None:
        wandb_run.finish()

    print(
        "\nTraining finished."
    )

    print(
        f"Metrics saved to: {metrics_path}"
    )

    print(
        f"Model saved to: {final_dir}"
    )


if __name__ == "__main__":
    main()