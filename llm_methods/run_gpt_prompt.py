from llm_methods.gpt_structure import *
from llm_methods.model import (
    speaker_generate_response,
    listener_retrieval_response,
    listener_select_response,
    speaker_retrieval_response,
    select_feature_response,
)

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

def run_gpt_prompt_wo_vocab(prompt,
                            token_limit:int):
    def create_prompt(prompt):
        if isinstance(prompt, str):
            with open("./llm_methods/prompt/sys_prompt_v1.txt","r",encoding="utf-8") as f:
                sys_prompt = f.read()
                sys_prompt=sys_prompt.replace("!<INPUT 0>!",str(token_limit))
            return {"system": sys_prompt, "user": prompt}
        elif isinstance(prompt, dict):
            return prompt
        else:
            raise ValueError("Invalid prompt format")
    prompt = create_prompt(prompt)
    
    def __func_validate(gpt_response,
                        prompt=None):
        try:
            if __func_clean_up(gpt_response, token_limit):
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def __func_clean_up(gpt_response,
                        prompt=None,
                        token_limit=token_limit):
        try:
            gpt_response=json.loads(gpt_response)["message"].strip().replace(",","").replace(".","")
            length , _ =tokens_check(gpt_response)
            return gpt_response if length <= token_limit else False
        except:
            return False

    def get_fail_safe():
        fs = "Error:The output of GPT is illegal."
        return fs
    
    fail_safe=get_fail_safe()
    
    gpt_param = {
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }
    
    output = safe_generate_response(
        prompt, 
        gpt_param, 
        12, 
        fail_safe, 
        __func_validate, 
        __func_clean_up
    )
    
    return output

def run_gpt_prompt_with_vocab(prompt,
                              token_limit:int,
                              token_vocab:dict):
    def create_prompt(prompt):
        if isinstance(prompt, str):
            with open("./llm_methods/prompt/sys_prompt_v2.txt","r",encoding="utf-8") as f:
                sys_prompt = f.read()
                sys_prompt=sys_prompt.replace("!<INPUT 0>!",str(token_limit)).replace("!<INPUT 1>!",json.dumps(token_vocab))
            return {"system": sys_prompt, "user": prompt}
        elif isinstance(prompt, dict):
            return prompt
        else:
            raise ValueError("Invalid prompt format")
    prompt = create_prompt(prompt)
    
    def __func_validate(gpt_response,
                        prompt=None):
        try:
            if __func_clean_up(gpt_response, token_limit):
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def __func_clean_up(gpt_response,
                        prompt=None,
                        token_limit=token_limit,
                        token_vocab=token_vocab):
        try:
            gpt_response=json.loads(gpt_response)["message"]
            length , token_list =tokens_check(gpt_response)
            if (length > token_limit) or (not set(token_list).issubset(set(token_vocab.keys()))):
                return False
            else:
                value_list=[]
                for num in token_list:
                    value_list.append(token_vocab[num])
                gpt_response=" ".join(value_list)
                return gpt_response
                    
        except:
            return False

    def get_fail_safe():
        fs = "Error:The output of GPT is illegal."
        return fs
    
    fail_safe=get_fail_safe()
    
    gpt_param = {
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }
    
    output = safe_generate_response(
        prompt, 
        gpt_param, 
        12, 
        fail_safe, 
        __func_validate, 
        __func_clean_up
    )
    
    return output

@safe_generate_response(
    response_model=speaker_generate_response,
    repeat=10,
    gpt_param=GPTPromptConfig(),
)
def run_gpt_prompt_speaker_generate(letters_count,
                              letters_list,
                              vocab,
                              obj_properties,
                              player_id,
                              failed_records,
                              max_length,
                              model,
                              verbose=True):
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
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": player_id,
        "max_length": max_length,
        "target_object": str(obj_properties),
        "letter_list": letters_list,
    }

@safe_generate_response(
    response_model=listener_retrieval_response,
    repeat=10,
    gpt_param=GPTPromptConfig(),
)
def run_gpt_prompt_listener_retrieval(letters_count,
                              max_words,
                              letters_list,
                              vocab,
                              given_word,
                              player_id,
                              model,
                              verbose=True):
    prompt_template = "prompt/listener_retrieval.txt"
    input_list = [
        str(letters_count),
        str(max_words),
        list_to_table(letters_list, "Letter"),
        list_to_table(vocab, "Word"),
        given_word,
    ]
    prompt = generate_prompt(input_list, prompt_template)
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": player_id,
        "target_word": given_word,
        "vocab": vocab,
    }

@safe_generate_response(
    response_model=listener_select_response,
    repeat=5,
    gpt_param=GPTPromptConfig(),
)
def run_gpt_prompt_listener_selection(letters_count,
                              letters_list,
                              vocab,
                              given_word,
                              semantic_features,
                              player_id,
                              model,
                              verbose=True):
    prompt_template = "prompt/listener_selection.txt"
    input_list = [
        str(letters_count),
        list_to_table(letters_list, "Letter"),
        dict_list_to_str(vocab),
        given_word,
        json.dumps(semantic_features, indent=4),
    ]
    prompt = generate_prompt(input_list, prompt_template)
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": player_id,
        "target_object": "",
        "word": given_word,
        "choices": semantic_features,
    }

@safe_generate_response(
    response_model=select_feature_response,
    repeat=5,
    gpt_param=GPTPromptConfig(),
)
def run_gpt_prompt_select_feature(word,
                              verbose=False):
    prompt_template = "prompt/select_features.txt"
    input_list = [word]
    prompt = generate_prompt(input_list, prompt_template)
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
    }

def _build_speaker_retrieval_prompt(object_features, features_list):
    prompt_template = "prompt/speaker_retrieval.txt"
    all_features_str = ""
    for i in range(len(features_list)):
        all_features_str += f"{i}:" + str(features_list[i]) + "\n"
    input_list = [str(object_features), all_features_str]
    prompt = generate_prompt(input_list, prompt_template)
    return prompt, prompt_template


@safe_generate_response(
    response_model=speaker_retrieval_response,
    repeat=10,
    gpt_param=GPTPromptConfig(),
)
def run_gpt_prompt_speaker_retrieval(
                              object_features,
                              features_list:list,
                              model,
                              verbose=True):
    prompt, prompt_template = _build_speaker_retrieval_prompt(object_features, features_list)
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": "",
        "max_length": 0,
        "target_object": "",
        "object_num": len(features_list),
    }


@safe_generate_response(
    response_model=speaker_retrieval_response,
    repeat=10,
    gpt_param=GPTPromptConfig(),
)
async def run_gpt_prompt_speaker_retrieval_async(
                              object_features,
                              features_list:list,
                              model,
                              verbose=True):
    prompt, prompt_template = _build_speaker_retrieval_prompt(object_features, features_list)
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "model": model,
        "player_id": "",
        "max_length": 0,
        "target_object": "",
        "object_num": len(features_list),
    }
