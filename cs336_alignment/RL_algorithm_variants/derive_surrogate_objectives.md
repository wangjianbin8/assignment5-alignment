这题其实就是把上一节的 **token-level surrogate policy** 从“每次只换 1 个 token”改成：

\[
\boxed{\text{每次连续换 2 个 token}}
\]

所以思路完全一样。

假设 response 长度为 \(L\)，并且 \(L\) 是偶数。题目的 estimator 按 pair 分成：

\[
(1,2),\ (3,4),\ (5,6),\ldots
\]

对于第 \(t\) 个 pair，对应的位置是：

\[
2t-1,\quad 2t
\]

---

## 1. 先回忆 token-level

上一节 token-level 的 surrogate policy \(\tilde\pi_t\) 是：

> 除了第 \(t\) 个位置使用当前 policy \(\pi_\theta\)，其他位置全部使用旧 policy \(\pi_0\)。

现在 pairwise 就自然变成：

> 除了第 \(2t-1\) 和 \(2t\) 两个位置使用 \(\pi_\theta\)，其他位置全部使用 \(\pi_0\)。

我们把它记成：

\[
\tilde\pi_t^{\text{pair}}
\]

---

## 2. 写出这个 surrogate policy

对于第 \(t\) 个 pair：

\[
\boxed{
\begin{aligned}
\tilde\pi_t^{\text{pair}}(y|x)
={}&
\left(
\prod_{s=1}^{2t-2}
\pi_0(y_s|x,y_{<s})
\right)
\\
&\cdot
\pi_\theta(y_{2t-1}|x,y_{<2t-1})
\\
&\cdot
\pi_\theta(y_{2t}|x,y_{<2t})
\\
&\cdot
\left(
\prod_{s=2t+1}^{L}
\pi_0(y_s|x,y_{<s})
\right).
\end{aligned}
}
\]

直观上就是：

```text
positions:

1 ... 2t-2 | 2t-1   2t | 2t+1 ... L
     π0       πθ     πθ       π0
```

注意：第二个 token \(y_{2t}\) 的 context 已经包含第一个由 \(\pi_\theta\) 产生的 \(y_{2t-1}\)，这是正常的。

---

# 3. 所以 surrogate objective 是什么？

和上一节完全平行：

\[
\boxed{
J_\theta^{\text{pair}}
=
E_{x\sim\rho}
\left[
\sum_{t=1}^{L/2}
E_{y\sim\tilde\pi_t^{\text{pair}}(\cdot|x)}
[r(y|x)]
\right]
}
\]

这就是最终要找的 surrogate objective。

意思是：

> 对每一对 token，假设只有这两个位置使用当前 policy，其余位置仍然使用旧 policy，然后计算 expected reward；最后把所有 pair 的 expected reward 加起来。

---

# 4. 为什么它对应题目的 estimator？

我们一步一步证明。

对于某一个 pair \(t\)，考虑：

\[
E_{y\sim\tilde\pi_t^{\text{pair}}}[r(y|x)]
\]

但是我们的真实数据是从：

\[
y\sim\pi_0
\]

采出来的，所以用 importance sampling 改写。

需要计算：

\[
\frac{
\tilde\pi_t^{\text{pair}}(y|x)
}{
\pi_0(y|x)
}
\]

---

## 5. 展开这个 ratio

旧 policy 整条 sequence：

\[
\pi_0(y|x)
=
\prod_{s=1}^L
\pi_0(y_s|x,y_{<s})
\]

而 surrogate policy 只有两个位置换成了 \(\pi_\theta\)。

所以两者一除，其他所有位置全部约掉，只剩：

\[
\boxed{
\frac{
\pi_\theta(y_{2t-1}|x,y_{<2t-1})
\pi_\theta(y_{2t}|x,y_{<2t})
}{
\pi_0(y_{2t-1}|x,y_{<2t-1})
\pi_0(y_{2t}|x,y_{<2t})
}
}
\]

这正好就是题目式 (55) 里的 importance weight。

---

# 6. 因此 surrogate objective 可以改写成

\[
J_\theta^{\text{pair}}
=
E_x
\left[
\sum_{t=1}^{L/2}
E_{y\sim\pi_0}
\left[
\frac{
\pi_\theta(y_{2t-1}|\cdots)
\pi_\theta(y_{2t}|\cdots)
}{
\pi_0(y_{2t-1}|\cdots)
\pi_0(y_{2t}|\cdots)
}
r(y|x)
\right]
\right]
\]

接下来对 \(\theta\) 求梯度。

---

# 7. 对两个 \(\pi_\theta\) 的乘积求梯度

定义：

\[
q_\theta
=
\pi_\theta(y_{2t-1}|\cdots)
\pi_\theta(y_{2t}|\cdots)
\]

利用我们之前一直用的 log derivative trick：

\[
\nabla_\theta q_\theta
=
q_\theta\nabla_\theta\log q_\theta
\]

所以：

\[
\nabla_\theta
[
\pi_\theta(y_{2t-1}|\cdots)
\pi_\theta(y_{2t}|\cdots)
]
\]

等于：

\[
\pi_\theta(y_{2t-1}|\cdots)
\pi_\theta(y_{2t}|\cdots)
\nabla_\theta
\log
[
\pi_\theta(y_{2t-1}|\cdots)
\pi_\theta(y_{2t}|\cdots)
]
\]

于是：

\[
\boxed{
\begin{aligned}
\nabla_\theta J_\theta^{\text{pair}}
=
E_xE_{y\sim\pi_0}
\Bigg[
\sum_{t=1}^{L/2}
&
\frac{
\pi_\theta(y_{2t-1}|\cdots)
\pi_\theta(y_{2t}|\cdots)
}{
\pi_0(y_{2t-1}|\cdots)
\pi_0(y_{2t}|\cdots)
}
\\
&\times r(y|x)
\nabla_\theta
\log
\left(
\pi_\theta(y_{2t-1}|\cdots)
\pi_\theta(y_{2t}|\cdots)
\right)
\Bigg].
\end{aligned}
}
\]

这正好就是题目给你的 estimator (55)。

---

# 8. 最直观的理解

前面的 token-level：

```text
pair/token 1:
πθ π0 π0 π0 ...

token 2:
π0 πθ π0 π0 ...

token 3:
π0 π0 πθ π0 ...
```

而 pairwise：

```text
pair 1:
πθ πθ π0 π0 π0 π0 ...

pair 2:
π0 π0 πθ πθ π0 π0 ...

pair 3:
π0 π0 π0 π0 πθ πθ ...
```

所以它处在：

\[
\text{token-level}
\longleftrightarrow
\text{sequence-level}
\]

之间。

token-level 每次只修正 1 个位置；

pairwise 每次修正 2 个位置；

sequence-level 修正整个 trajectory。

---

## 作业 Deliverable 可以这样写

\[
\boxed{
J_\theta^{\mathrm{pair}}
=
\mathbb E_{x\sim\rho}
\left[
\sum_{t=1}^{L/2}
\mathbb E_{y\sim\tilde\pi_t^{\mathrm{pair}}(\cdot|x)}
[r(y|x)]
\right]
}
\]

where

\[
\tilde\pi_t^{\mathrm{pair}}(y|x)
=
\left(\prod_{s<2t-1}\pi_0(y_s|x,y_{<s})\right)
\pi_\theta(y_{2t-1}|x,y_{<2t-1})
\pi_\theta(y_{2t}|x,y_{<2t})
\left(\prod_{s>2t}\pi_0(y_s|x,y_{<s})\right).
\]

> The pairwise surrogate policy samples all positions from the stale policy \(\pi_0\), except for positions \(2t-1\) and \(2t\), which are sampled from the current policy \(\pi_\theta\). Rewriting the expectation under \(\tilde\pi_t^{\mathrm{pair}}\) as an expectation under \(\pi_0\) gives the importance ratio
> \[
> \frac{\pi_\theta(y_{2t-1}|\cdots)\pi_\theta(y_{2t}|\cdots)}
> {\pi_0(y_{2t-1}|\cdots)\pi_0(y_{2t}|\cdots)}.
> \]
> Applying the log-derivative trick to the product of the two current-policy probabilities produces exactly the estimator in Eq. (55).

核心记忆：

\[
\boxed{\text{pairwise estimator = 每次把两个 timestep 从 }\pi_0\text{ 替换成 }\pi_\theta\text{ 的 surrogate objective。}}
\]