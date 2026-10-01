import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

#这个函数其实是计算所有位置的log probabilities
#后面 GRPO loss 会使用response_mask把不属于 response 的位置消掉。
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits
    all_log_probs = F.log_softmax(logits, dim=-1)

    '''
    对每个位置，只取真实 label token 对应的
    log probability
    '''
    token_log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {"log_probs": token_log_probs}

    if return_token_entropy:
        probs = all_log_probs.exp()
        token_entropy = -(probs * all_log_probs).sum(dim=-1)

        result["token_entropy"] = token_entropy

    return result