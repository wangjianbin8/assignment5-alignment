这道题是这一章最核心的数学推导之一。它要你回答：

> **不同 advantage normalization，其实隐含地给不同难度 prompt 加了什么权重？**

也就是找：

\[
w(x)
\]

使得：

\[
\nabla_\theta J_{\theta,w}
=
\nabla_\theta
E_x[
w(x)
E_y[r(y|x)]
]
\]

等价于对应 GRPO estimator。

---

我们一步一步推。

---

# 0. 先定义一个重要变量：prompt 难度

令：

\[
\eta(x)
=
E_{y\sim\pi_\theta}
[r(y|x)]
\]

什么意思？

就是：

> 当前模型面对 prompt \(x\)，平均 reward 是多少。

例如：

简单题：

\[
\eta(x)=0.9
\]


困难题：

\[
\eta(x)=0.1
\]


---

注意 binary reward：

\[
r\in\{0,1\}
\]

所以：

\[
\eta(x)
\]

其实就是：

\[
P(\text{correct}|x)
\]

也就是正确率。

---

# 1. 目标形式

我们希望找到：

\[
w(x)
\]


使：

\[
\nabla_\theta J_{\theta,w}
\]

成立。


定义：

\[
J_{\theta,w}
=
E_x[
w(x)
E_y[r(y|x)]
]
\]


因为：

\[
w
\]

不求梯度：

(stopgrad)


所以：

\[
\nabla_\theta J_{\theta,w}
=
E_x[
w(x)
\nabla_\theta
E_y[r]
]
\]


使用 policy gradient：

\[
=
E_x[
w(x)
E_y[
r\nabla\log\pi(y|x)
]
]
\]


所以最终形式：

\[
\boxed{
E_x[
w(x)
E_y[
r\nabla\log\pi
]
]
}
\]


我们只需要比较每个算法的 estimator。

---

# (a) Dr. GRPO

题目：

\[
E_x[
\frac1G
\sum_{j=1}^G
(r_j-\mu)
\nabla\log\pi_j
]
\]


令：

\[
G\rightarrow\infty
\]


那么：

group mean：

\[
\mu
\]

收敛：

\[
\mu\rightarrow\eta(x)
\]


所以：

\[
r_j-\mu
\]

变成：

\[
r-\eta(x)
\]


因此 estimator：

\[
E_x[
E_y[
(r-\eta(x))
\nabla\log\pi
]
]
\]


展开：

\[
=
E_x[
E_y[
r\nabla\log\pi
]
-
\eta(x)
E_y[
\nabla\log\pi
]
]
\]


第二项：

\[
E_y[
\nabla\log\pi
]=0
\]


所以：

\[
=
E_x[
E_y[
r\nabla\log\pi
]
]
\]


比较：

\[
E_x[
w(x)
E_y[
r\nabla\log\pi
]
]
\]


得到：

\[
\boxed{
w(x)=1
}
\]


---

## 解释

Dr. GRPO：

\[
A=r-\mu
\]


虽然减了 baseline：

\[
\mu
\]

但是 baseline 项期望为 0。

所以：

它没有改变不同 prompt 的权重。

所有题：

一样重要。


答案：

\[
\boxed{w_{Dr.GRPO}(x)=1}
\]

---

# (b) Standard GRPO

现在：

\[
A=
\frac{r-\mu}{std}
\]


Estimator：

\[
E_x[
\frac1G
\sum_j
\frac{r_j-\mu}{std}
\nabla\log\pi_j
]
\]


当：

\[
G\rightarrow\infty
\]


有：

\[
\mu\rightarrow\eta(x)
\]


并且：

\[
std\rightarrow\sigma(x)
\]


其中：

\[
\sigma(x)
=
\sqrt{
Var(r|x)
}
\]


所以：

\[
=
E_x[
\frac1{\sigma(x)}
E_y[
(r-\eta(x))
\nabla\log\pi
]
]
\]


刚才证明：

\[
E[
(r-\eta)
\nabla\log\pi
]
=
E[
r\nabla\log\pi
]
\]


所以：

\[
=
E_x[
\frac1{\sigma(x)}
E_y[
r\nabla\log\pi
]
]
\]


比较：

\[
w(x)
E_y[r\nabla\log\pi]
\]


得到：

\[
\boxed{
w(x)=\frac1{\sigma(x)}
}
\]


---

## 进一步化简

binary reward：

\[
r\in\{0,1\}
\]


所以：

\[
Var(r)
=
p(1-p)
\]


这里：

\[
p=\eta(x)
\]


因此：

\[
\sigma(x)
=
\sqrt{\eta(x)(1-\eta(x))}
\]


所以：

\[
\boxed{
w(x)
=
\frac1{\sqrt{\eta(x)(1-\eta(x))}}
}
\]


---

## 直觉

如果：

\[
\eta=0.5
\]

std 最大：

\[
\sigma=0.5
\]


weight:

\[
2
\]


如果：

\[
\eta=0.99
\]


std:

\[
0.1
\]


weight:

\[
10
\]


所以：

Standard GRPO 更强调：

- 太简单的问题
- 太难的问题


因为它们 reward variance 小。

---

# (c) MaxRL

MaxRL：

\[
A=
\frac{r-\mu}{\mu}
\]


同样：

\[
\mu\rightarrow\eta(x)
\]


所以：

\[
=
E_x[
\frac1{\eta(x)}
E_y[
(r-\eta)
\nabla\log\pi
]
]
\]


利用之前：

\[
E[(r-\eta)\nabla\log\pi]
=
E[r\nabla\log\pi]
\]


得到：

\[
=
E_x[
\frac1{\eta(x)}
E_y[
r\nabla\log\pi
]
]
\]


因此：

\[
\boxed{
w(x)=\frac1{\eta(x)}
}
\]


---

# 最终答案表

| 方法 | advantage | reweight \(w(x)\) |
|-|-|-|
| Dr. GRPO | \(r-\mu\) | \(\boxed{1}\) |
| GRPO | \(\frac{r-\mu}{std}\) | \(\boxed{\frac1{\sigma(x)}}\) |
| MaxRL | \(\frac{r-\mu}{\mu}\) | \(\boxed{\frac1{\eta(x)}}\) |

---

# 直觉总结（非常重要）

## Dr. GRPO

\[
w(x)=1
\]

所有 prompt：

一样训练。


---

## GRPO

\[
w(x)=
\frac1{\sqrt{\eta(1-\eta)}}
\]


奖励方差小的 prompt 权重大。


---

## MaxRL

\[
w(x)=
\frac1{\eta}
\]


困难 prompt：

\[
\eta低
\]


权重大。


所以：

\[
\boxed{
\text{MaxRL = 自动增加困难题的训练权重}
}
\]


这就是这一节最后一句：

> dividing by average reward will on average upweight more difficult prompts

的数学证明。你前面学的 baseline、advantage normalization、Dr.GRPO，现在全部串起来了。