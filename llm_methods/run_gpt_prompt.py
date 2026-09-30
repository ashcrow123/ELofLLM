from llm_methods.gpt_structure import *
from llm_methods.model import (
    speaker_generate_response,
    listener_retrieval_response,
    listener_select_response,
    speaker_retrieval_response,
)
from typing import Tuple
def list_to_table(lst,obj):
    if not lst:
        raise ValueError("列表不能为空")

    # 编号宽度（根据最大编号长度来自动对齐）
    index_width = len(str(len(lst)))
    # 内容宽度（最长的元素字符串）
    content_width = max(len(str(item)) for item in lst)

    # 构造表格行
    lines = []
    header = f"{'Serial Number'.ljust(index_width)} | {obj.ljust(content_width)}"
    divider = '-' * len(header)
    lines.append(header)
    lines.append(divider)

    for idx, item in enumerate(lst, 1):
        lines.append(f"{str(idx).ljust(index_width)} | {str(item).ljust(content_width)}")

    return '\n'.join(lines)

def dict_list_to_str(dict_list):
    
    if not dict_list:
        return ""

    if not all(isinstance(d, dict) for d in dict_list):
        raise ValueError("All elements in the list must be dictionaries. ")

    return '\n'.join(str(d) for d in dict_list)


@safe_generate_response(response_model=speaker_generate_response)
def run_gpt_prompt_speaker_generate(letters_count,
                              letters_list,
                              vocab,
                              obj_properties,
                              player_id,
                              failed_records,
                              max_length,
                              model,
                              verbose=True)-> Tuple[dict,GPTPromptConfig]:
    prompt_template = "prompt/speaker_generate.txt"
    input_list = [
        str(letters_count),
        list_to_table(letters_list, "Letter"),
        dict_list_to_str(vocab),
        str(obj_properties),
        json.dumps(failed_records),
        str(max_length),
    ]
    prompt = generate_prompt(input_list, prompt_template)
    gpt_param=GPTPromptConfig(
        model=model,
        max_tokens= 4096,
        top_p= 1.0,
        frequency_penalty= 0.0,
        presence_penalty= 0.0,
        temperature= 0.1
    )
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": player_id,
        "max_length": max_length,
        "target_object": str(obj_properties),
        "letter_list": letters_list,
    },gpt_param

@safe_generate_response(response_model=listener_retrieval_response)
def run_gpt_prompt_listener_retrieval(letters_count,
                              max_words,
                              letters_list,
                              vocab,
                              given_word,
                              player_id,
                              model,
                              verbose=True)-> Tuple[dict,GPTPromptConfig]:
    prompt_template = "prompt/listener_retrieval.txt"
    input_list = [
        str(letters_count),
        str(max_words),
        list_to_table(letters_list, "Letter"),
        list_to_table(vocab, "Word"),
        given_word,
    ]
    prompt = generate_prompt(input_list, prompt_template)
    gpt_param=GPTPromptConfig(
        model=model,
        max_tokens= 4096,
        top_p= 1.0,
        frequency_penalty= 0.0,
        presence_penalty= 0.0,
        temperature= 0.1
    )
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": player_id,
        "target_word": given_word,
        "vocab": vocab,
    },gpt_param

@safe_generate_response(response_model=listener_select_response)
def run_gpt_prompt_listener_selection(letters_count,
                              letters_list,
                              vocab,
                              given_word,
                              semantic_features,
                              player_id,
                              model,
                              verbose=True)-> Tuple[dict,GPTPromptConfig]:
    prompt_template = "prompt/listener_selection.txt"
    input_list = [
        str(letters_count),
        list_to_table(letters_list, "Letter"),
        dict_list_to_str(vocab),
        given_word,
        json.dumps(semantic_features, indent=4),
    ]
    prompt = generate_prompt(input_list, prompt_template)
    gpt_param=GPTPromptConfig(
        model=model,
        max_tokens= 4096,
        top_p= 1.0,
        frequency_penalty= 0.0,
        presence_penalty= 0.0,
        temperature= 0.1
    )
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": player_id,
        "target_object": "",
        "word": given_word,
        "choices": semantic_features,
    },gpt_param


def _build_speaker_retrieval_prompt(object_features, features_list):
    prompt_template = "prompt/speaker_retrieval.txt"
    all_features_str = ""
    for i in range(len(features_list)):
        all_features_str += f"{i}:" + str(features_list[i]) + "\n"
    input_list = [str(object_features), all_features_str]
    prompt = generate_prompt(input_list, prompt_template)
    return prompt, prompt_template


@safe_generate_response(response_model=speaker_retrieval_response)
def run_gpt_prompt_speaker_retrieval(
                              object_features,
                              features_list:list,
                              model,
                              verbose=True)-> Tuple[dict,GPTPromptConfig]:
    prompt, prompt_template = _build_speaker_retrieval_prompt(object_features, features_list)
    gpt_param=GPTPromptConfig(
        model=model,
        max_tokens= 4096,
        top_p= 1.0,
        frequency_penalty= 0.0,
        presence_penalty= 0.0,
        temperature= 0.1
    )
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": "",
        "max_length": 0,
        "target_object": "",
        "object_num": len(features_list),
    },gpt_param


@safe_generate_response(response_model=speaker_retrieval_response)
async def run_gpt_prompt_speaker_retrieval_async(
                              object_features,
                              features_list:list,
                              model,
                              verbose=True)-> Tuple[dict,GPTPromptConfig]:
    prompt, prompt_template = _build_speaker_retrieval_prompt(object_features, features_list)
    gpt_param=GPTPromptConfig(
        model=model,
        max_tokens= 4096,
        top_p= 1.0,
        frequency_penalty= 0.0,
        presence_penalty= 0.0,
        temperature= 0.1
    )
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": "",
        "max_length": 0,
        "target_object": "",
        "object_num": len(features_list),
    },gpt_param
