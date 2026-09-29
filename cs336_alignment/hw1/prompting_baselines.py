import json 

# vLLM 推理服务器：
# 负责加载模型，并根据 prompt 生成回答
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer 

# reward function：
# 根据模型输出和标准答案计算奖励
# r1_zero_reward_fn:
#   用于 <think>...</think><answer>...</answer> 格式
# question_only_reward_fn:
#   用于 \boxed{} 格式
from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn
)


# 三种实验使用的 prompt 模板文件
# 对应作业要求：
# 1. zero-shot question_only
# 2. zero-shot r1_zero
# 3. few-shot r1_zero_three_shot
PROMPT_PATHS = { 
    "question_only":
        "cs336_alignment/prompts/question_only.prompt", 

    "r1_zero":
        "cs336_alignment/prompts/r1_zero.prompt", 

    "r1_zero_three_shot":
        "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt", 
}

def load_gsm8k(path): 
    data = []

    # GSM8K 是 jsonl 文件
    # 每一行都是一个 JSON 样本
    with open(path, "r") as f: 
        for line in f: 
            data.append(json.loads(line))
    return data

def extract_answer(answer):
    return answer.split("####")[-1].strip()

#加载提示词模板
def load_prompt(path):
    with open(path, "r") as f:
        return f.read()

#生成提示词
def build_prompts(data, prompt_type):
    template = load_prompt(PROMPT_PATHS[prompt_type])
    prompts = []
    for item in data:
        question = item["question"]
        prompt = template.format(question = question)
        prompts.append(prompt)
    return prompts

def get_sampling_params(prompt_type):
    params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        # vLLM 需要：
        # 每个 prompt 生成几个回答
        "n": 1,
        "seed": 0,
    }

    if prompt_type in [
        "r1_zero",
        "r1_zero_three_shot"
    ]:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True

    return params

#进行评分
def compute_rewards(prompt_type, responses, ground_truths, ):
    if prompt_type == "question_only":
        reward_fn = question_only_reward_fn
    else:
        reward_fn = r1_zero_reward_fn

    results = []
    for response, answer in zip(responses, ground_truths):
        result = reward_fn(response, answer)
        results.append(result)
    return results

#按题目要求统计category
def summarize_rewards(results, prompts, responses, ground_truths):
    category1 = 0
    category2 = 0
    category3 = 0

    category2_examples = []
    category3_examples = []

    for i, r in enumerate(results):
        item = {
            "prompt": prompts[i],
            "response": responses[i],
            "ground_truth": ground_truths[i],
            "reward": r,
        }

        if r["format_reward"] == 1 and r["answer_reward"] == 1: 
            category1 += 1
        elif r["format_reward"] == 1 and r["answer_reward"] == 0:
            category2 += 1
            if len(category2_examples) < 10:
                category2_examples.append(item)

        elif r["format_reward"] == 0 and r["answer_reward"] == 0:
            category3 += 1
            if len(category3_examples) < 10:
                category3_examples.append(item)

    return {
        "category1": category1,
        "category2": category2,
        "category3": category3,   
        "category2_examples": category2_examples,
        "category3_examples": category3_examples,
    }

server = VLLMServer(
    model_id="allenai/OLMo-2-0425-1B",
    gpu=0,
    seed=0,
    gpu_memory_utilization=0.75,
)

def evaluate(prompt_type):
    #加载数据
    data = load_gsm8k("data/gsm8k/test.jsonl")[:20]
    #生成提示词
    prompts = build_prompts(data, prompt_type)
    #生成答案
    params = get_sampling_params(prompt_type)
    completions = server.generate_completions(prompts, params, batch_size=2)
    responses = [c.text for c in completions]

    #和正确答案比较计算
    ground_truths = [extract_answer(item["answer"]) for item in data]
    rewards = compute_rewards(prompt_type, responses, ground_truths)

    #统计category
    metrics = summarize_rewards(rewards, prompts, responses, ground_truths,)
    return {
        "metrics": metrics,

        # 保存几个案例
        # 用于作业报告分析模型行为
        "examples": list(zip(prompts[:5], responses[:5]))
    }



def main():
    server.start()

    for prompt_type in [
        "question_only",
        "r1_zero",
        "r1_zero_three_shot",
    ]:

        res = evaluate(prompt_type)

        print(prompt_type)
        print(res["metrics"]["category1"])
        print(res["metrics"]["category2"])
        print(res["metrics"]["category3"])

        print("\nCategory2 examples")
        for example in res["metrics"]["category2_examples"]:
            print(example)

        print("\nCategory3 examples")
        for example in res["metrics"]["category3_examples"]:
            print(example)

if __name__ == "__main__":
    main()

'''
GSM8K question
        |
        v
Prompt construction
        |
        v
OLMo-2-0425-1B
        |
        v
Generated response y
        |
        v
Reward function
        |
        v
category1/2/3
'''