这道题是在比较：

\[
\boxed{\text{RFT}}
\]

和

\[
\boxed{\text{Dr. GRPO}}
\]

两个目标函数。

核心问题：

1. 它们是不是优化同一个东西？
2. 期望（expectation）一样吗？
3. 谁方差更低？
4. 什么情况下用哪个？

我们一步一步分析。

---

# 1. 先写两个公式

## RFT objective

题目：

\[
J_{\theta}^{RFT}
=
\frac1Z
\sum_x
\sum_{j=1}^G
\mathbf1\{r(y^{(j)}|x)=1\}
\log\pi_\theta(y^{(j)}|x)
\]


因为 reward 是 binary：

\[
r\in\{0,1\}
\]


所以：

\[
\mathbf1\{r=1\}=r
\]


因此 RFT 可以写：

\[
\boxed{
J_{RFT}
=
\frac1Z
\sum_j
r(y^{(j)}|x)
\log\pi_\theta(y^{(j)}|x)
}
\]


梯度：

\[
\nabla J_{RFT}
=
\frac1Z
\sum_j
r(y^{(j)}|x)
\nabla\log\pi_\theta(y^{(j)}|x)
\]


---

## Dr. GRPO

公式：

\[
\hat g_{DrGRPO}
=
\frac1Z
\sum_j
(r(y^{(j)}|x)-\mu)
\nabla\log\pi_\theta(y^{(j)}|x)
\]

其中：

\[
\mu
=
\frac1G\sum_j r(y^{(j)}|x)
\]


所以：

\[
\boxed{
A_j=r_j-\mu
}
\]


---

# 2. 最大区别在哪里？

RFT：

\[
\boxed{
weight=r_j
}
\]


Dr. GRPO：

\[
\boxed{
weight=r_j-\mu
}
\]


也就是：

## RFT：

只看绝对正确/错误。

例如：

reward:

\[
[1,0,0,0]
\]


权重：

\[
[1,0,0,0]
\]


结果：

```text
正确答案 ↑
错误答案 不管
```


---

## Dr. GRPO：

看相对组内表现。

同样：

\[
[1,0,0,0]
\]


mean:

\[
\mu=0.25
\]


advantage：

\[
[0.75,-0.25,-0.25,-0.25]
\]


结果：

```text
正确答案 ↑
错误答案 ↓
```


所以：

\[
\boxed{
RFT只奖励好的
}
\]

\[
\boxed{
GRPO奖励好的，同时惩罚差的
}
\]


---

# 3. 它们 expectation 一样吗？

这是重点。

---

## RFT

梯度：

\[
E[
r\nabla\log\pi
]
\]


这就是 REINFORCE：

\[
\boxed{
\nabla J
}
\]


所以 RFT 的期望：

\[
\boxed{
E[g_{RFT}]
=
\nabla J
}
\]


---

## Dr. GRPO

梯度：

\[
E[
(r-\mu)
\nabla\log\pi
]
\]


展开：

\[
=
E[
r\nabla\log\pi
]
-
E[
\mu\nabla\log\pi
]
\]


第一项：

\[
=
\nabla J
\]


第二项呢？


关键：

\[
\mu
\]

是 group mean：

\[
\mu=\frac1G\sum r_j
\]


它依赖其他 sample。

不是普通 constant baseline。

---

但是根据之前 baseline 推导：

如果 baseline 不依赖当前 action：

\[
E[b\nabla\log\pi]=0
\]


这里：

\[
\mu
\]

包含当前 sample 的 reward。

所以：

\[
\boxed{
Dr.GRPO 不完全保持原始 policy gradient expectation
}
\]


不过当：

\[
G
\]

很大时：

一个 sample 对 mean 的影响很小。

于是：

\[
\mu
\]

接近一个独立 baseline。


所以：

\[
\boxed{
G\rightarrow\infty}
\]

时，两者越来越接近。

---

# 4. 谁 variance 更低？

通常：

\[
\boxed{
Dr.GRPO variance 更低
}
\]


原因：

RFT：

假设：

\[
r\in\{0,1\}
\]


如果：

大部分答案错误：

比如：

\[
[0,0,0,1]
\]


RFT：

只有一个样本产生梯度：

```text
正确:
↑↑↑

错误:
0
```


梯度非常稀疏。


---

Dr.GRPO：

mean:

\[
\mu=0.25
\]


weights：

\[
[-0.25,-0.25,-0.25,0.75]
\]


所有 sample 都提供信息。


所以：

\[
\boxed{
利用更多 rollout 信息
}
\]


variance 通常更低。

---

# 5. 什么时候 RFT 更合适？

RFT：

适合：

### reward 非常可靠

比如：

数学题：

```
答案正确 = 1
错误 = 0
```

grader 很准确。


并且：

模型已经有一定能力。


例如：

采样：

```text
8个回答

7个正确
1个错误
```

那么：

RFT 得到很多高质量 SFT 数据。

---

# 6. 什么时候 Dr.GRPO 更好？

当：

模型很弱。


例如：

8个回答：

```text
0 0 0 0 0 0 0 1
```


RFT：

只有一个有效训练样本。


Dr.GRPO：

告诉模型：

```
这个最好
其他7个比平均差
```

所以学习信号更多。

---

# 7. 一个直观比较表

| | RFT | Dr.GRPO |
|-|-|-|
| 权重 | \(r\) | \(r-\mu\) |
| 正确答案 | 增加概率 | 增加概率 |
| 错误答案 | 忽略 | 降低概率 |
| 是否用相对信息 | ❌ | ✅ |
| 方差 | 较高 | 通常较低 |
| 是否严格 policy gradient | binary reward 下接近 | group baseline 引入偏差 |
| 类似 | SFT | RL |

---

# 8. 作业 deliverable 可以写：

> RFT and Dr. GRPO differ in how they weight sampled responses. For binary rewards, RFT uses the reward itself as the weight, keeping only correct responses and performing supervised fine-tuning on them. Dr. GRPO instead uses the centered reward \(r-\mu\), which increases the probability of above-average responses and decreases the probability of below-average responses. RFT has the same expectation as the REINFORCE policy gradient under on-policy sampling, while Dr. GRPO introduces a group-dependent baseline that may slightly change the expectation. However, Dr. GRPO typically has lower variance because it uses information from both successful and unsuccessful samples. RFT may be preferable when the reward signal is reliable and many correct samples can be obtained, while Dr. GRPO may be preferable when successful rollouts are sparse.

这个答案基本就是这道题想考的核心。