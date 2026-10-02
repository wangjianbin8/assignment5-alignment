对，这一小节就是把 **Standard GRPO → Dr. GRPO** 的两个变化真正写进我们前面的函数。我们按题目顺序来。

## 1. `think_about_length_normalization`

可以这样理解两种做法。

### Standard GRPO：每条 sequence 除自己的长度

\[
L_i
=
\frac{1}{|y_i|}
\sum_t \ell_{i,t}
\]

优点是：**每条 response 大致具有相同权重**。一个 20-token 回答和一个 200-token 回答，不会仅仅因为后者 token 更多就产生 10 倍大的梯度，训练通常更稳定。

缺点是它改变了原始 policy gradient。原始 sequence log-probability 本来是

\[
\log \pi(y|x)
=
\sum_t \log \pi(y_t|x,y_{<t})
\]

里面没有 \(1/|y|\)。因此 sequence normalization 会相对放大短回答、缩小长回答。

---

### Dr. GRPO：统一除固定常数

\[
L
=
\frac1Z
\sum_i\sum_t\ell_{i,t}
\]

所有 token 使用同一个 normalization。

优点是更接近原始 policy-gradient estimator，不会因为某条 response 自己比较长就额外把它缩小。

缺点是长 response 包含更多 token，因此自然会贡献更大的梯度，可能导致训练方差更大，也可能让 response length 对更新幅度产生明显影响。

一个比较合适的作业答案可以写：

> Sequence-length normalization gives every response approximately equal weight regardless of its length, which can improve optimization stability and prevent long responses from dominating the gradient. However, it changes the original policy-gradient estimator because the log-probability of a trajectory is naturally a sum over tokens rather than an average. Constant normalization preserves this token-sum structure more faithfully, but long responses can contribute larger gradients and therefore increase variance or introduce length-dependent effects. Sequence normalization may be preferable when response lengths vary substantially and equal weighting of examples is desired, while constant normalization is preferable when faithfully approximating the original policy gradient is the priority.
