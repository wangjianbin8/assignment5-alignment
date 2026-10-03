这题就是总结我们刚才学的 **bias–variance tradeoff**。

可以先记成：

| 方法 | Bias | Variance | 直觉 |
|---|---|---|---|
| No reweighting | 最大 | 最低 | 完全不修正 stale policy |
| PPO/GRPO token-level clipping | 中等 | 较低 | 每个 token 单独修正并 clipping |
| GSPO sequence-level geometric mean | 较小/更接近 sequence-level | 较高 | 用整条 sequence 的信息，但几何平均和 clipping 控制方差 |

为什么这样排？不做 importance reweighting 时，我们直接把 \(\pi_0\) 生成的数据当成 \(\pi_\theta\) 的数据，所以当两个 policy 差得比较远时 bias 最大，但也不会引入 importance ratio 的额外波动，因此 variance 最小。

PPO/GRPO 的 token-level 方法：

\[
\rho_t=\frac{\pi_\theta(y_t|\cdots)}{\pi_0(y_t|\cdots)}
\]

能够部分修正 distribution mismatch，而且 clipping 限制极端 ratio，因此 variance 较低；但因为 prefix 和 suffix 仍然按照旧 policy 处理，所以仍然有 bias。

GSPO：

\[
s=
\left(\prod_t\rho_t\right)^{1/L}
\]

利用整条 sequence 的 ratio 信息，因此直觉上比 token-level 更接近 sequence-level correction；但它仍然因为 geometric mean 和 clipping 而有 bias，同时 sequence-level weight 会让不同 sequence 之间的梯度差异更大，因此 variance 通常比 token-level 更高。

### 作业 Deliverable 可以写

> No importance reweighting lies at the low-variance, high-bias end of the spectrum, since stale samples are used without correcting for the difference between \(\pi_0\) and \(\pi_\theta\). PPO/GRPO-style clipped token-level reweighting provides a middle ground: it partially corrects the distribution mismatch while clipping keeps variance relatively low, but it remains biased because prefixes and suffixes are not fully reweighted. GSPO uses information from the whole sequence and may therefore reduce some of this token-level bias, but its sequence-level weights can introduce more variance; the geometric mean and clipping are used to control this. No reweighting may work well when \(\pi_\theta\) remains very close to \(\pi_0\), token-level clipping is attractive for stable training with long responses, while GSPO may be preferable when sequence-level dependencies are important and some additional variance is acceptable.