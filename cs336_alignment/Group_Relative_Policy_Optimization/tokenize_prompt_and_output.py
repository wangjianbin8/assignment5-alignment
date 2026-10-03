import torch
from transformers import PreTrainedTokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:

    ids_list = []
    masks_list = []


    for prompt, output in zip(prompt_strs, output_strs):

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False
        ).input_ids

        output_ids = tokenizer(
            output,
            add_special_tokens=False
        ).input_ids


        # prompt + response
        ids = prompt_ids + output_ids


        # 对齐完整ids
        mask = (
            [0] * len(prompt_ids)
            +
            [1] * len(output_ids)
        )


        ids_list.append(ids)
        masks_list.append(mask)



    max_len = max(
        len(ids)
        for ids in ids_list
    )


    pad_id = tokenizer.pad_token_id


    padded_ids = []
    padded_masks = []

    for ids, mask in zip(
        ids_list,
        masks_list
    ):

        padding = max_len - len(ids)

        padded_ids.append(
            ids + [pad_id] * padding
        )

        padded_masks.append(
            mask + [0] * padding
        )


    padded_ids = torch.tensor(
        padded_ids,
        dtype=torch.long
    )

    padded_masks = torch.tensor(
        padded_masks,
        dtype=torch.long
    )


    return {
        # remove last token
        "input_ids": padded_ids[:, :-1],

        # remove first token
        "labels": padded_ids[:, 1:],

        # align with labels
        "response_mask": padded_masks[:, 1:],
    }