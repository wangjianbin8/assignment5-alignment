这题就是比较三种 advantage：

\[
\text{GRPO: }\frac{r-\mu}{\mathrm{std}},
\qquad
\text{MaxRL: }\frac{r-\mu}{\mu},
\qquad
\text{Dr. GRPO: }r-\mu
\]

核心区别是：**它们会让不同 prompt 对梯度贡献不同的权重。**

### 1. 除以 group std：Standard GRPO

\[
A=\frac{r-\mu}{\mathrm{std}}
\]

优点是把不同 group 的 advantage 尺度拉得比较接近，通常有利于优化稳定性。

缺点是它会改变原始 policy-gradient 的 weighting，而且当 group std 很小时，除法可能放大噪声，因此需要 \(\epsilon\)。

根据上一题，对于 binary reward：

\[
w(x)\propto
\frac{1}
{\sqrt{\eta(x)(1-\eta(x))}}
\]

所以它并不是简单地对所有 prompt 等权。

---

### 2. 除以 group mean：MaxRL

\[
A=\frac{r-\mu}{\mu}
\]

因为困难题的平均正确率 \(\mu\) 较低，所以：

\[
\frac1\mu
\]

更大。

例如：

\[
\mu_{\text{easy}}=0.8
\Rightarrow
1/\mu=1.25
\]

而：

\[
\mu_{\text{hard}}=0.1
\Rightarrow
1/\mu=10
\]

所以 MaxRL 会：

\[
\boxed{\text{更强调当前模型比较不会做的题}}
\]

优点是可以把学习能力集中到困难 prompt。

缺点是当 \(\mu\) 很小时，更新可能非常大、方差更高；如果 reward 有噪声，偶然成功的困难题也可能被过度放大，因此同样需要 \(\epsilon\)。

---

### 3. 不做 advantage normalization：Dr. GRPO

\[
A=r-\mu
\]

优点是最简单，而且更接近我们原本的 expected-reward policy gradient；在 \(G\to\infty\) 的推导下：

\[
w(x)=1
\]

也就是没有额外的 difficulty reweighting。

缺点是不同 group 的 advantage scale 不会被人为统一，所以优化上可能没有 std normalization 那么稳定。

---

### 怎么选？

可以这样记：

| 方法 | 主要作用 | 潜在问题 |
|---|---|---|
| std | 稳定不同 group 的 gradient scale | 改变目标权重；std 小时可能放大噪声 |
| mean | 强调困难 prompt | mean 很小时梯度可能很大 |
| none | 更忠实原始 policy gradient | 缺少额外尺度控制 |

如果不同 group 的 reward scale/variance 差异很大、主要关心训练稳定性，std normalization 可能有优势；如果希望特别学习当前较困难的题，mean normalization（MaxRL）很自然；如果 reward 本身尺度稳定，并希望尽量保持原始 policy-gradient 的含义，那么不做 normalization（Dr. GRPO）比较合理。

### 作业 Deliverable 可以写

> Standard-deviation normalization makes the scale of advantages more comparable across groups, which may improve optimization stability, but it changes the prompt weighting induced by the original policy gradient and can amplify noise when the group standard deviation is small. Mean normalization, as in MaxRL, upweights prompts with low average reward and therefore focuses training on harder examples, but it may produce large or noisy updates when the mean reward is close to zero. Using no advantage normalization, as in Dr. GRPO, is more faithful to the original expected-reward policy gradient and avoids these extra reweightings, although its gradient scale may vary more across prompts. Thus std normalization may be useful for stability, mean normalization when emphasizing difficult prompts is desirable, and no normalization when preserving the original policy-gradient objective is the priority.