# CS336 Assignment 5 - Problem (baseline_calcs) 解题过程

## (a) 无 baseline 的 policy gradient variance

策略：

\(\pi_\theta(A=1)=p=\sigma(\theta)\)
奖励：

\(r(A)=\mathbb{1}_{A=1}\)

因此：

- \(A=1\)：概率 p，reward \(=1\)
- \(A=0\)：概率 \(1-p\)，reward \(=0\)

定义单个样本：

\(Z=r(A)\nabla_\theta\log\pi_\theta(A)\)

因为：

\(Var(\hat g)=\frac{1}{n}Var(Z)\)
所以先求 Z。

### 求梯度项

当 \(A=1\)：

\(\nabla_\theta\log p = \frac{1}{p}\frac{dp}{d\theta}\)

sigmoid 导数：

\(\frac{dp}{d\theta}=p(1-p)\)

所以：

\(\nabla_\theta\log p=1-p\)

当 \(A=0\)：

\(\nabla_\theta\log(1-p) = \frac{-p(1-p)}{1-p}\)

所以：

\(\nabla_\theta\log(1-p)=-p\)

因此：

\(Z=
\begin{cases}
1-p,&\text{概率 }p\\
0,&\text{概率 }1-p
\end{cases}\)期望：

\(E[Z]=p(1-p)\)

二阶矩：

\(E[Z^2]=p(1-p)^2\)

方差：

\(Var(Z)=E[Z^2]-E[Z]^2\)

得到：

\(Var(Z)=p(1-p)^3\)

因此：

\(\boxed{Var(\hat g)=\frac{p(1-p)^3}{n}}\)

---

# (b) 加入 baseline

定义：

\(Z_b=(r(A)-b)\nabla_\theta\log\pi_\theta(A)\)

当 \(A=1\)：

\(Z_b=(1-b)(1-p)\)
概率：p

当 \(A=0\)：

\(Z_b=bp\)
概率：\(1-p\)

所以：

\(Z_b=
\begin{cases}
(1-b)(1-p),&p\\
bp,&1-p
\end{cases}\)期望：

\(E[Z_b] = p(1-b)(1-p)+(1-p)bp\)

化简：

\(\boxed{E[Z_b]=p(1-p)}\)

说明 baseline 不改变期望。

二阶矩：

\(E[Z_b^2] = p(1-b)^2(1-p)^2 + (1-p)b^2p^2\)

因此：

\(Var(Z_b) = E[Z_b^2]-E[Z_b]^2\)

化简：

\(\boxed{Var(Z_b)=p(1-p)(1-p-b)^2}\)

n 个样本：

\(\boxed{Var(\hat g_b)=\frac{p(1-p)(1-p-b)^2}{n}}\)

---

# (c) population mean baseline

reward 的平均值：

\(E[r(A)] = p\)

所以：

\(b=p\)

代入：

\(\boxed{Var(\hat g_p)=\frac{p(1-p)(1-2p)^2}{n}}\)

比较：
无 baseline：

\(\frac{p(1-p)^3}{n}\)

baseline：

\(\frac{p(1-p)(1-2p)^2}{n}\)

比较 \((1-2p)^2\) 和 \((1-p)^2\)：
得到：

\((1-2p)^2<(1-p)^2\)

等价于：

\(p<\frac23\)

## 最终结论

- 当 \(p<\frac23\)：baseline 降低 variance
- 当 \(p=\frac23\)：variance 相同
- 当 \(p>\frac23\)：baseline 增加 variance

核心：

\(\boxed{\text{baseline保持梯度期望不变，但是改变方差}}\)

> 
> 这也是 GRPO 使用 group relative advantage 的数学基础。