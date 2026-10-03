# Problem (prompting_baselines): Run OLMo-2-0425-1B on GSM8K

## (a) Prompting baseline evaluation

本实验使用 OLMo-2-0425-1B 模型，在 GSM8K 数据集上测试三种 prompt：

1.  zero-shot question_only
2.  zero-shot r1_zero
3.  few-shot r1_zero_three_shot

生成参数：

-   temperature = 1.0
-   top-p = 1.0
-   maximum generation length = 512

r1_zero 和 r1_zero_three_shot 使用 `</answer>` 作为停止字符串。

------------------------------------------------------------------------

## Evaluation metrics

Category 定义：

-   Category 1: format reward = 1 且 correctness reward = 1
-   Category 2: format reward = 1 且 correctness reward = 0
-   Category 3: format reward = 0 且 correctness reward = 0

  Prompt                 Category 1   Category 2   Category 3
  -------------------- ------------ ------------ ------------
  question_only                   0            5           15
  r1_zero                         0            4           16
  r1_zero_three_shot              1           15            4

------------------------------------------------------------------------

## Examples and observations

### question_only

question_only 只提供问题，并要求最终答案放入 `\\boxed{}`。

观察发现模型经常生成额外文本，而不是专注解决当前数学问题。例如模型可能生成
boxed answer 后继续生成其他无关内容。

这说明仅提供问题不足以让 base model 稳定进入数学解题模式。

### r1_zero

r1_zero 要求模型输出：

    <think>
    reasoning
    </think>
    <answer>
    answer
    </answer>

模型更容易遵守格式，并产生 reasoning 过程。

但是格式正确不代表答案正确，很多输出仍然包含错误推理或错误答案。

### r1_zero_three_shot

few-shot prompt 提供三个示例，使模型可以模仿目标输出格式。

实验中 Category 3 明显减少，说明 few-shot prompting 提高了格式遵循能力。

但是它并不能保证数学答案正确。

------------------------------------------------------------------------

## Category 2 analysis

Category 2 表示模型输出格式正确，但是答案错误。

人工检查多个样例后发现，大部分 Category 2 都是真实数学错误，而不是
parser 解析失败。

因此 Category 2 主要反映模型推理能力不足。

------------------------------------------------------------------------

## Category 3 analysis

Category 3 表示模型没有遵循要求格式。

例如模型可能直接输出：

    The answer is 72.

而不是：

    <answer>
    72
    </answer>

即使人类能够理解答案，grader 仍然无法解析。

------------------------------------------------------------------------

# (b) Characterizing model behavior

## question_only

question_only 对模型行为约束较弱。

模型可能生成无关文本、继续训练数据中的内容，而不是解决当前问题。

## r1_zero

r1_zero 通过明确要求 reasoning 和 answer
标签，将模型行为转向类似数学助手。

它提高了格式遵循能力，但不能保证数学推理正确。

## r1_zero_three_shot

few-shot 示例进一步强化了模型对目标格式的模仿。

模型更稳定地产生 `<think>` 和 `<answer>` 结构，大幅减少格式错误。

但是 few-shot 主要改善输出形式，而不是直接提升数学能力。

------------------------------------------------------------------------

# Conclusion

Prompting 可以显著改变 base model 的行为模式：

-   question_only 约束较弱；
-   r1_zero 能诱导 reasoning 风格输出；
-   r1_zero_three_shot 能通过示例提高格式遵循能力。

但是 prompt 本身不能替代模型已有的推理能力。
