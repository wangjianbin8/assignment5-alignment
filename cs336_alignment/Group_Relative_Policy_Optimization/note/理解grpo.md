可以。把这条链真正搞懂之后，后面的 GRPO training loop 基本就通了：

\[
\boxed{
\text{reward}
\rightarrow
\text{advantage}
\rightarrow
\text{token log-prob}
\rightarrow
\text{per-token loss}
\rightarrow
\text{scalar loss}
\rightarrow
\text{backward}
}
\]

我们从最开始的目标一步一步推到代码，不跳步。

---

# 1. 最开始到底想优化什么？

强化学习的目标非常直接：

> 模型生成答案，希望答案的平均 reward 越高越好。

讲义写的是：

\[
J_\theta
=
\mathbb E_{x\sim\rho}
\mathbb E_{y\sim\pi_\theta(y|x)}
[r(y|x)]
\]

其中：

- \(x\)：题目
- \(y\)：模型生成的完整回答
- \(\rho\)：题目分布
- \(\pi_\theta\)：当前模型
- \(r(y|x)\)：回答得到的 reward
- \(\theta\)：模型参数

也就是说：

\[
\boxed{\text{目标：最大化 expected reward}}
\]

:chatgpt-content-reference{index="0"}

---

# 2. 为什么不能直接对 reward 求梯度？

例如模型生成：

```text
<answer>42</answer>
```

grader 给：

\[
r=1
\]

生成错了：

\[
r=0
\]

问题是：

\[
r
\]

本身不是神经网络的可微函数。

你不能直接：

```python
reward.backward()
```

因为 grader 可能只是字符串比较：

```python
if answer == ground_truth:
    return 1
else:
    return 0
```

那怎么办？

我们需要通过：

\[
\pi_\theta(y|x)
\]

也就是：

> 模型生成这个回答的概率

来间接调整模型。

---

# 3. REINFORCE 是怎么来的？

先固定一个题目 \(x\)。

目标：

\[
J(\theta)
=
\mathbb E_{y\sim\pi_\theta}[r(y)]
\]

这里“期望”展开就是：

\[
J(\theta)
=
\sum_y
\pi_\theta(y|x)r(y|x)
\]

意思是：

> 所有可能回答的 reward，按模型生成它们的概率加权平均。

现在对 \(\theta\) 求梯度：

\[
\nabla_\theta J
=
\nabla_\theta
\sum_y
\pi_\theta(y|x)r(y|x)
\]

reward 不依赖 \(\theta\)，所以：

\[
=
\sum_y
r(y|x)
\nabla_\theta\pi_\theta(y|x)
\]

使用前面学过的 log-derivative trick：

\[
\nabla_\theta \pi_\theta
=
\pi_\theta
\nabla_\theta\log\pi_\theta
\]

所以：

\[
\nabla_\theta J
=
\sum_y
r(y|x)
\pi_\theta(y|x)
\nabla_\theta
\log\pi_\theta(y|x)
\]

重新写成 expectation：

\[
\boxed{
\nabla_\theta J
=
\mathbb E_{y\sim\pi_\theta}
\left[
r(y|x)
\nabla_\theta\log\pi_\theta(y|x)
\right]
}
\]

这就是 REINFORCE。:chatgpt-content-reference{index="1"}

先记住最核心结构：

\[
\boxed{
reward
\times
\nabla\log probability
}
\]

---

# 4. reward 怎么变成 advantage？

直接用 reward：

\[
r\nabla\log\pi
\]

是可以的。

但是 GRPO 不直接这么做。

它让**同一道题生成 \(G\) 个答案**。

例如：

```text
题目 x₁

response 1 → reward 1
response 2 → reward 0
response 3 → reward 1
response 4 → reward 0
```

所以：

\[
[r_{11},r_{12},r_{13},r_{14}]
=
[1,0,1,0]
\]

---

## 4.1 先计算 group mean

\[
\mu_1
=
\frac{1}{G}
\sum_{j=1}^G r_{1j}
\]

这里：

\[
\mu_1
=
\frac{1+0+1+0}{4}
=
0.5
\]

然后：

\[
r_{ij}-\mu_i
\]

得到：

\[
[0.5,-0.5,0.5,-0.5]
\]

这已经非常有意义：

```text
reward = 1
→ 比同组平均好
→ positive advantage

reward = 0
→ 比同组平均差
→ negative advantage
```

所以 advantage 可以先理解为：

\[
\boxed{
\text{这个回答比同一道题的其他回答好多少}
}
\]

baseline 减法在期望上利用了 score-function identity；讲义在 baseline 部分专门说明了这一点。:chatgpt-content-reference{index="2"}

---

# 5. GRPO 还会除以 std

标准 GRPO：

\[
\boxed{
A_{ij}
=
\frac{r_{ij}-\mu_i}
{\sigma_i+\epsilon}
}
\]

所以：

```python
group_means = rewards.mean(dim=1)
group_stds = rewards.std(dim=1)

advantages = (
    rewards - group_means
) / (
    group_stds + eps
)
```

为什么除 std？

主要是 normalization：

> 不让不同 group 的 advantage scale 差得太大。

所以到这里：

```text
raw reward
    ↓
减 group mean
    ↓
除 group std
    ↓
advantage
```

即：

\[
\boxed{
r_{ij}
\rightarrow
A_{ij}
}
\]

---

# 6. 接下来为什么突然出现 log probability？

现在对于某个完整回答 \(y\)：

\[
\pi_\theta(y|x)
\]

表示：

> 模型生成完整回答 \(y\) 的概率。

但是语言模型是一 token 一 token 生成的。

假设：

\[
y=(y_1,y_2,y_3)
\]

根据概率链式法则：

\[
\pi_\theta(y|x)
=
\pi_\theta(y_1|x)
\pi_\theta(y_2|x,y_1)
\pi_\theta(y_3|x,y_1,y_2)
\]

更一般：

\[
\boxed{
\pi_\theta(y|x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t|x,y_{<t})
}
\]

---

# 7. 为什么必须取 log？

因为：

\[
\log(ab)=\log a+\log b
\]

所以：

\[
\log\pi_\theta(y|x)
=
\log
\prod_t
\pi_\theta(y_t|x,y_{<t})
\]

变成：

\[
\boxed{
\log\pi_\theta(y|x)
=
\sum_t
\log\pi_\theta(y_t|x,y_{<t})
}
\]

这一步非常重要。

因为现在完整 sequence 的 log probability，被拆成了：

> 每一个 token 的 log probability。

所以我们的 `get_response_log_probs` 返回：

```text
(BG, T)
```

例如某条回答：

```text
token 1 → -0.2
token 2 → -0.4
token 3 → -0.1
```

对应：

\[
[
\log p(y_1|x),
\log p(y_2|x,y_1),
\log p(y_3|x,y_{<3})
]
\]

---

# 8. advantage 怎么和 token log-prob 连起来？

原来的 policy gradient：

\[
A
\nabla_\theta
\log\pi_\theta(y|x)
\]

而：

\[
\log\pi_\theta(y|x)
=
\sum_t
\log\pi_\theta(y_t|x,y_{<t})
\]

所以：

\[
A
\nabla_\theta
\sum_t
\log\pi_\theta(y_t|\cdots)
\]

因为求导和求和可以交换：

\[
=
\sum_t
A
\nabla_\theta
\log\pi_\theta(y_t|\cdots)
\]

于是每个 token 都贡献：

\[
\boxed{
A
\nabla_\theta
\log\pi_\theta(y_t|\cdots)
}
\]

这就是为什么我们需要 **per-token log probability**。:chatgpt-content-reference{index="3"}

---

# 9. 但是 PyTorch 需要 loss，不是 gradient

现在我们希望得到：

\[
A
\nabla_\theta\log\pi_\theta
\]

我们可以构造：

\[
J_{\text{token}}
=
A\log\pi_\theta
\]

因为：

\[
\nabla_\theta
J_{\text{token}}
=
A\nabla_\theta\log\pi_\theta
\]

注意 \(A\) 在当前 backward 中被当成常数。

---

但是我们想：

\[
\max J
\]

而 PyTorch optimizer 是：

\[
\min L
\]

所以令：

\[
L=-J
\]

因此：

\[
\boxed{
\ell_{i,t}
=
-A_i
\log\pi_\theta(y_{i,t}|\cdots)
}
\]

这就是：

```python
per_token_policy_gradient_loss = (
    -advantages * policy_log_probs
)
```

---

# 10. 为什么 advantage 为正会提高概率？

这一点一定要真正理解。

假设：

\[
A=+1
\]

那么：

\[
L=-\log p
\]

optimizer 要让 loss 下降。

例如：

原来：

\[
p=0.2
\]

那么：

\[
-\log0.2\approx1.61
\]

如果概率提高到：

\[
p=0.8
\]

那么：

\[
-\log0.8\approx0.22
\]

loss 下降。

所以：

\[
A>0
\Rightarrow
p\uparrow
\]

也就是：

> 好回答 → 增加以后生成这个回答的概率。

---

# 11. advantage 为负呢？

假设：

\[
A=-1
\]

那么：

\[
L
=
-(-1)\log p
=
\log p
\]

optimizer 要让：

\[
\log p
\]

越来越小。

例如：

\[
p:0.8\rightarrow0.2
\]

那么：

\[
\log0.8=-0.22
\]

变成：

\[
\log0.2=-1.61
\]

变得更小。

因此：

\[
A<0
\Rightarrow
p\downarrow
\]

也就是：

> 差回答 → 降低以后生成它的概率。

---

# 12. 为什么是 per-token loss？

假设一条 response：

```text
answer token 1
answer token 2
answer token 3
```

log probs：

\[
[-0.2,-0.5,-0.3]
\]

advantage：

\[
A=0.8
\]

那么每个 token：

\[
\ell_t
=
-0.8\times\log p_t
\]

得到：

\[
[0.16,0.40,0.24]
\]

所以现在还是：

```text
sequence 1:
[0.16, 0.40, 0.24]
```

还不是最终 loss。

---

# 13. 为什么还要 response mask？

我们的 tensor 不只有 response。

可能是：

```text
prompt prompt prompt response response response pad
```

mask：

```text
0      0      0      1        1        1       0
```

所以：

\[
m_{it}\in\{0,1\}
\]

真正参与训练的是：

\[
m_{it}\ell_{it}
\]

例如：

```text
loss:
[0.7, 0.8, 0.6, 0.16, 0.40, 0.24, 0.5]

mask:
[0,   0,   0,   1,    1,    1,    0]
```

相乘：

```text
[0, 0, 0, 0.16, 0.40, 0.24, 0]
```

这样 prompt 和 padding 不参与 GRPO。

---

# 14. 为什么最后还要 aggregate？

现在还有一个问题：

```python
per_token_loss.shape
```

是：

\[
(BG,T)
\]

但是：

```python
loss.backward()
```

一般需要最终构造一个 scalar objective。

standard GRPO 的方式是：

## 先对一条 sequence 的 response token 平均

对于第 \(k\) 条 rollout：

\[
L_k
=
\frac{
\sum_t m_{kt}\ell_{kt}
}{
\sum_t m_{kt}
}
\]

注意：

\[
\sum_t m_{kt}
\]

其实就是：

\[
\operatorname{len}(y_k)
\]

所以：

\[
\boxed{
L_k
=
\frac1{|y_k|}
\sum_{t=1}^{|y_k|}
\ell_{kt}
}
\]

---

# 15. 然后 sequence 之间平均

一共有：

\[
BG
\]

条 rollout。

因此：

\[
\boxed{
L
=
\frac1{BG}
\sum_{k=1}^{BG}L_k
}
\]

把前面的式子代进去：

\[
L
=
-\frac1{BG}
\sum_{i=1}^{B}
\sum_{j=1}^{G}
\frac1{|y^{(i,j)}|}
\sum_t
A_{ij}
\log
\pi_\theta
(y_t^{(i,j)}|x^{(i)},y_{<t}^{(i,j)})
\]

这就是我们代码最终构造的 scalar loss。

---

# 16. 对这个 scalar loss 求梯度会发生什么？

对：

\[
L
=
-\frac1{BG}
\sum_{ij}
\frac1{|y_{ij}|}
\sum_t
A_{ij}
\log\pi_\theta(y_{ijt}|\cdots)
\]

求梯度：

\[
\nabla_\theta L
=
-\frac1{BG}
\sum_{ij}
\frac1{|y_{ij}|}
\sum_t
A_{ij}
\nabla_\theta
\log\pi_\theta(y_{ijt}|\cdots)
\]

然后 PyTorch gradient descent：

\[
\theta
\leftarrow
\theta-\alpha\nabla_\theta L
\]

代入：

\[
\theta
\leftarrow
\theta
+
\alpha
\frac1{BG}
\sum_{ij}
\frac1{|y_{ij}|}
\sum_t
A_{ij}
\nabla_\theta
\log\pi_\theta(y_{ijt}|\cdots)
\]

看见了吗？

负负得正。

最终正好变成 Algorithm 1 里的 **gradient ascent**：

\[
\boxed{
\theta
\leftarrow
\theta
+
\alpha\hat g
}
\]

---

# 17. 现在把整个代码 pipeline 对上数学

### ① rollout

vLLM：

\[
y^{(i,j)}
\sim
\pi_\theta(y|x^{(i)})
\]

---

### ② reward

```python
compute_rollout_rewards(...)
```

得到：

\[
r_{ij}
\]

shape：

```text
(BG,)
```

---

### ③ advantage

```python
compute_group_normalized_rewards(...)
```

计算：

\[
A_{ij}
=
\frac{
r_{ij}-\mu_i
}{
\sigma_i+\epsilon
}
\]

shape：

```text
(BG,)
```

---

### ④ token log probability

```python
get_response_log_probs(...)
```

得到：

\[
\log\pi_\theta
(y_{ijt}|x_i,y_{ij,<t})
\]

shape：

```text
(BG, T)
```

---

### ⑤ per-token policy gradient loss

```python
compute_policy_gradient_loss(...)
```

计算：

\[
\boxed{
\ell_{ijt}
=
-A_{ij}
\log\pi_\theta(y_{ijt}|\cdots)
}
\]

shape：

```text
(BG, T)
```

---

### ⑥ mask + sequence aggregation

```python
aggregate_loss_across_microbatch(...)
```

先：

\[
L_{ij}
=
\frac{
\sum_t m_{ijt}\ell_{ijt}
}{
\sum_t m_{ijt}
}
\]

再：

\[
L
=
\frac1{BG}
\sum_{ij}L_{ij}
\]

最后：

```text
scalar
```

---

### ⑦ backward

```python
loss.backward()
```

得到：

\[
\nabla_\theta L
\]

---

### ⑧ optimizer step

```python
optimizer.step()
```

因为：

\[
L=-J
\]

gradient descent on \(L\)

等价于：

gradient ascent on \(J\)。

---

# 18. 用一句人话把整个 GRPO 说完

假设同一道题生成 4 个答案：

```text
A: reward 1
B: reward 0
C: reward 1
D: reward 0
```

GRPO 先问：

> 谁比组内平均表现好？

于是：

```text
A → positive advantage
B → negative advantage
C → positive advantage
D → negative advantage
```

然后看这些回答实际上由哪些 token 组成。

对于 A：

```text
token a1
token a2
token a3
```

因为 A 是好回答：

\[
A>0
\]

训练就增加：

\[
P(a_1),P(a_2|a_1),P(a_3|a_1,a_2)
\]

对于 B，因为：

\[
A<0
\]

就降低组成 B 的这些 token 在对应上下文里的概率。

所以 GRPO 本质就是：

\[
\boxed{
\text{比同组平均好的生成轨迹，提高概率；
比同组平均差的生成轨迹，降低概率。}
}
\]

而数学上实现这个机制的关键就是：

\[
\boxed{
-A\log\pi_\theta
}
\]

这一项。

最后你可以把整条链记成：

\[
\boxed{
r
\rightarrow
A
\rightarrow
\log\pi
\rightarrow
-A\log\pi
\rightarrow
\text{token平均}
\rightarrow
\text{sequence平均}
\rightarrow
L
\rightarrow
\nabla_\theta L
}
\]
