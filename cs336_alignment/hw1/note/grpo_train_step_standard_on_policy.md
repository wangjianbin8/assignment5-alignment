这一题就是把前面写的所有零件真正组装成一次 **完整的 GRPO 参数更新**。先不要急着看代码，先把执行顺序固定下来：

\[
\boxed{
\text{rollouts}
\rightarrow
\text{reward}
\rightarrow
\text{advantage}
\rightarrow
\text{分 microbatch}
\rightarrow
\log\pi
\rightarrow
-A\log\pi
\rightarrow
\text{聚合 loss}
\rightarrow
backward
\rightarrow
\text{clip grad}
\rightarrow
optimizer.step()
}
\]

最重要的一点是：**reward 和 advantage 要先在整个 rollout batch 上计算，之后才拆 microbatch。**因为 advantage 要比较同一道题的 \(G\) 个回答，不能随便把 group 拆散后各算各的。

---

# 1. 输入到底是什么？

假设：

\[
B=2,\qquad G=4
\]

也就是 2 道题，每题生成 4 个回答。

那么：

```python
len(repeated_prompts) == 8
len(rollout_responses) == 8
len(repeated_ground_truths) == 8
```

数据逻辑上是：

```text
prompt 1 ─ response 1
         ├ response 2
         ├ response 3
         └ response 4

prompt 2 ─ response 5
         ├ response 6
         ├ response 7
         └ response 8
```

所以：

\[
N = BG = 8
\]

这就是 rollout batch size。

---

# 2. 第一步：计算 raw rewards

调用我们前面写的：

```python
raw_rewards, reward_metadata = compute_rollout_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
)
```

得到：

```text
raw_rewards.shape = (N,)
```

例如：

\[
[1,0,1,0,\quad 0,0,1,1]
\]

对应：

```text
第一题4个回答    第二题4个回答
```

---

# 3. 第二步：把 reward 变成 advantage

调用：

```python
advantages, advantage_metadata = (
    compute_group_normalized_rewards(
        raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
)
```

标准 GRPO：

\[
A_{ij}
=
\frac{r_{ij}-\mu_i}
{\sigma_i+\epsilon}
\]

输出：

```text
advantages.shape = (N,)
```

每条 response 一个 advantage。

到这里还完全不需要跑大模型 forward，所以很省显存。

---

# 4. 第三步：tokenize 整个 rollout batch

调用第一题：

```python
train_batch = tokenize_prompt_and_output(
    repeated_prompts,
    rollout_responses,
    tokenizer,
)
```

得到：

```python
train_batch["input_ids"]
train_batch["labels"]
train_batch["response_mask"]
```

shape 都类似：

\[
(N,T)
\]

这里依然可以先放 CPU。

---

# 5. 为什么现在才拆 microbatch？

真正吃显存的是：

```python
model(input_ids)
```

因为 Hugging Face 模型需要保存 backward 所需的 activation。

假设：

```text
N = 128
gradient_accumulation_steps = 4
```

那么拆成：

```text
32
32
32
32
```

每次只让 32 条进 GPU 做 forward/backward。

---

# 6. 每个 microbatch 到底干什么？

对于一个 microbatch：

```text
input_ids_micro
labels_micro
mask_micro
advantages_micro
```

首先：

```python
response_log_probs = get_response_log_probs(
    model,
    input_ids_micro,
    labels_micro,
    return_token_entropy=True,
)
```

得到：

\[
\log\pi_\theta(y_t|\cdots)
\]

以及 entropy。

然后：

```python
per_token_loss, loss_metadata = (
    compute_policy_gradient_loss(
        advantages_micro,
        response_log_probs["log_probs"],
    )
)
```

数学：

\[
\ell_{it}
=
-A_i\log\pi_\theta(y_{it}|\cdots)
\]

然后：

```python
microbatch_loss = aggregate_loss_across_microbatch(
    per_token_loss,
    mask_micro,
)
```

得到这个 microbatch 的 sequence-normalized 平均 loss：

\[
L_m
=
\frac1{M_m}
\sum_{i\in m}
L_i
\]

---

# 7. 这里必须进行 gradient accumulation 修正

假设整个 batch 大小：

\[
N
\]

当前 microbatch 大小：

\[
M_m
\]

必须：

\[
\boxed{
\tilde L_m
=
L_m
\frac{M_m}{N}
}
\]

代码：

```python
scaled_loss = (
    microbatch_loss
    * microbatch_size
    / full_batch_size
)
```

然后：

```python
scaled_loss.backward()
```

每个 microbatch 都这样做。

为什么？

因为：

\[
L_m
=
\frac1{M_m}
\sum_{i\in m}L_i
\]

所以：

\[
\frac{M_m}{N}L_m
=
\frac1N
\sum_{i\in m}L_i
\]

所有 microbatch 加起来：

\[
\sum_m
\frac{M_m}{N}L_m
=
\frac1N
\sum_{i=1}^{N}L_i
\]

正好就是整个 batch 一次计算的 loss。

---

# 8. 所有 microbatch 做完后，才更新参数

循环中：

```python
scaled_loss.backward()
```

梯度一直累加。

等全部完成：

```python
optimizer.step()
```

才真正更新一次：

\[
\theta_{\text{old}}
\rightarrow
\theta_{\text{new}}
\]

因此：

\[
\boxed{
一个 rollout batch
=
一个 optimizer step
}
\]

---

# 9. 更新之前还要 gradient clipping

题目要求：

> clip the gradient norm to `max_grad_norm`

所以在：

```python
optimizer.step()
```

之前：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_grad_norm,
)
```

假设原来的整体梯度范数：

\[
\|\nabla_\theta L\|_2=20
\]

而：

```python
max_grad_norm = 1.0
```

那么 PyTorch 会整体缩放 gradient，使最终 norm 不超过 1。

这是为了防止偶尔出现特别大的梯度把训练搞崩。

而 `clip_grad_norm_` 的返回值是：

> **clipping 前的 gradient norm**

所以正好可以拿来 logging。

---

# 10. 一个标准实现

假设前面那些 helper function 都已经写好了，可以这样组装：

```python
import math
from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer


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
```

---

# 11. 这里有一个非常重要的“分界线”

整个 train step 可以分成两部分。

### 不吃显存、需要全 batch 信息

```text
compute_rollout_rewards
        ↓
compute_group_normalized_rewards
```

为什么要 whole batch？

因为：

\[
A_{ij}
\]

依赖同一 group 的所有 rewards。

---

### 吃显存、可以拆 microbatch

```text
get_response_log_probs
        ↓
compute_policy_gradient_loss
        ↓
aggregate_loss
        ↓
backward
```

因为这里需要真正跑 transformer forward/backward。

所以整个设计思想就是：

\[
\boxed{
\text{统计类操作 whole batch 做}
\qquad
\text{神经网络 forward/backward 分 microbatch 做}
}
\]

---

# 12. `token_entropy` 为什么只是 logging？

entropy：

\[
H(p)
=
-\sum_vp(v)\log p(v)
\]

它告诉我们模型 next-token distribution 有多不确定。

这里题目只是要求：

> log token entropy

所以我们并没有把 entropy 加入 loss。

也就是说：

```text
entropy
    ↓
metadata / wandb / logging
```

不是：

```text
entropy
    ↓
loss
```

这两件事情不要混淆。

---

# 13. 最终整个 train step 的数学形式

每条 rollout：

\[
A_i
\]

每个 token：

\[
\ell_{it}
=
-A_i
\log\pi_\theta(y_{it}|x_i,y_{i,<t})
\]

每条 response：

\[
L_i
=
\frac{
\sum_t m_{it}\ell_{it}
}{
\sum_t m_{it}
}
\]

整个 rollout batch：

\[
\boxed{
L
=
\frac1N
\sum_{i=1}^NL_i
}
\]

因为显存不够，我们没有一次计算 \(L\)，而是：

\[
L
=
\sum_m
\frac{M_m}{N}
L_m
\]

所以依次：

```python
scaled_micro_loss.backward()
```

最终 `.grad` 得到的依然是：

\[
\boxed{
\nabla_\theta L_{\text{full batch}}
}
\]

然后：

```python
clip_grad_norm_
optimizer.step()
```

完成一次 GRPO 更新。

这道 5 分题实际上没有引入太多新的数学，真正考的是：**你能不能把前面六个 helper 函数按正确顺序拼起来，并且让 microbatch gradient accumulation 与 full-batch GRPO 数学上等价。**

这道题现在就是把前面所有函数真正组装成一次完整的 **standard on-policy GRPO update**。

最值得你记住的是执行顺序：

\[
\boxed{
\text{rollouts}
\rightarrow
\text{rewards}
\rightarrow
\text{advantages}
\rightarrow
\text{tokenize}
\rightarrow
\text{microbatch forward}
\rightarrow
\log\pi
\rightarrow
-A\log\pi
\rightarrow
\text{sequence avg}
\rightarrow
\text{scaled backward}
\rightarrow
\text{clip grad}
\rightarrow
\text{optimizer.step()}
}
\]

前半段 `reward/advantage` 需要看整个 rollout batch；后半段真正跑模型的部分因为显存贵，所以拆成 microbatch。

你可以把它理解成下面这张程序结构：

```text
整个 rollout batch
│
├─ compute_rollout_rewards
│      ↓
│   raw_rewards
│
├─ compute_group_normalized_rewards
│      ↓
│   advantages
│
├─ tokenize_prompt_and_output
│      ↓
│   input_ids / labels / response_mask
│
└─ 拆成 microbatches
       │
       ├─ get_response_log_probs
       │      ↓
       │   log_probs + entropy
       │
       ├─ compute_policy_gradient_loss
       │      ↓
       │   -A * logπ   （per-token）
       │
       ├─ aggregate_loss_across_microbatch
       │      ↓
       │   microbatch scalar loss
       │
       ├─ × microbatch_size / full_batch_size
       │
       └─ backward()
       
所有 microbatch 完成
       ↓
clip_grad_norm_
       ↓
optimizer.step()
       ↓
optimizer.zero_grad()
```

数学上，每条 rollout \(i\) 有 advantage \(A_i\)，每个 response token 有：

\[
\ell_{i,t}
=
-A_i\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
\]

先在一条 response 内平均：

\[
L_i
=
\frac{1}{|y_i|}
\sum_t \ell_{i,t}
\]

整个 batch 的目标是：

\[
L
=
\frac1N\sum_{i=1}^N L_i
\]

但显存不够，所以第 \(m\) 个 microbatch 大小为 \(M_m\) 时，它自己算出的平均 loss 是：

\[
L_m
=
\frac1{M_m}
\sum_{i\in m}L_i
\]

因此 backward 前必须乘：

\[
\frac{M_m}{N}
\]

于是：

\[
\frac{M_m}{N}L_m
=
\frac1N
\sum_{i\in m}L_i
\]

所有 microbatch 的梯度累加后，正好等价于整个 batch 一次计算：

\[
\boxed{
\sum_m
\nabla_\theta
\left(
\frac{M_m}{N}L_m
\right)
=
\nabla_\theta
\left(
\frac1N\sum_iL_i
\right)
}
\]

这就是这道题最核心的地方。

实现时还要注意四个 logging 指标：`loss`、`grad_norm`、`token_entropy`、train reward（至少 total reward 和 format reward）。其中 entropy 只是监控，不加入当前 standard GRPO loss。

最后，在所有 microbatch 都 `backward()` 完之后：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_grad_norm,
)

optimizer.step()
optimizer.zero_grad()
```

`clip_grad_norm_` 放在 `optimizer.step()` 前，它返回的是裁剪前的总梯度范数，正好可以拿来记录。

所以这道 5 分题本质上没有新的核心公式，而是在考你能不能把前面写好的这些函数：

```text
tokenize_prompt_and_output
get_response_log_probs
compute_rollout_rewards
compute_group_normalized_rewards
compute_policy_gradient_loss
aggregate_loss_across_microbatch
```

按正确顺序拼起来，并且把 gradient accumulation 的缩放做对。