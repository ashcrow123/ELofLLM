from llm_methods.gpt_structure import *
def deal_json_format(text):
    text=text.replace("**EXPECTED FORMAT:**","")
    text="{"+text.split("{")[-1]
    text=text.replace('json',"")
    text=text.replace('`','')
    text=text.strip()
    return text

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

def split_cv_blocks(word):
    # """将单词分割成辅音+元音组合块"""
    # if "-" in word:
    #     word="".join(word.split("-"))
    # blocks = []
    # i = 0
    # while i < len(word) - 1:
    #     c, v = word[i], word[i+1]
    #     if c.lower() in "bcdfghjklmnpqrstvwxyz" and v.lower() in "aeiou":
    #         blocks.append(c + v)
    #         i += 2
    #     else:
    #         raise ValueError(f"Invalid CV block at position {i}: {c+v}")
    # if i < len(word): 
    #     raise ValueError(f"Word has leftover characters at the end: {word[i:]}")
    blocks=word.split("-")
    return blocks

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

def run_gpt_prompt_speaker_generate(letters_count,
                              letters_list,
                              vocab,
                              obj_properties,
                              player_id,
                              failed_records,
                              max_length,
                              model,
                              verbose=True):
    prompt_template="prompt/speaker_generate.txt"
    def create_prompt(letters_count,
                      letters_list,
                      vocab,
                      obj_properties,
                      failed_records,
                      prompt_template=prompt_template):
        
        input_list=[]
        input_list.append(str(letters_count))
        input_list.append(list_to_table(letters_list,"Letter"))
        input_list.append(dict_list_to_str(vocab))
        input_list.append(str(obj_properties))
        input_list.append(json.dumps(failed_records))
        input_list.append(str(max_length))
        prompt=generate_prompt(
            input_list,
            prompt_template
        )
        return prompt,input_list
    
    
    def __func_validate(gpt_response,
                        prompt=None):
        try:
            if __func_clean_up(gpt_response):
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def __func_clean_up(gpt_response,
                        letters_list=letters_list,
                        prompt=None,):
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            word=gpt_response.get("word","")
            if (not word) or (not isinstance(word, str)):
                return False

            blocks=split_cv_blocks(word)
            for letter in blocks:
                if letter not in letters_list:
                    return False
            if len(blocks)>=1 and len(blocks)<=max_length:
                # gpt_response["word"]=word
                return gpt_response
            else:
                return False
                    
        except:
            return False

    def get_fail_safe():
        fs = "Error:The output of GPT is illegal."
        return fs
    
    fail_safe=get_fail_safe()
    
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }
    prompt,input_list=create_prompt(letters_count,
                        letters_list,
                        vocab,
                        obj_properties,
                        failed_records)
    
    output = safe_generate_response(
        prompt, 
        gpt_param, 
        10, 
        fail_safe, 
        __func_validate, 
        __func_clean_up
    )
    if verbose:
        print_run_prompts(
            prompt_template=prompt_template,
            player_id=player_id,
            prompt=prompt,
            output=output,
        )
    
    return output

def run_gpt_prompt_select_resembling_words(letters_count,
                              max_words,
                              letters_list,
                              vocab,
                              given_word,
                              player_id,
                              model,
                              verbose=True):
    prompt_template="prompt/select_resembling_words.txt"
    def create_prompt(letters_count,
                      max_words,
                      letters_list,
                      vocab,
                      given_word,
                      prompt_template=prompt_template):
        
        input_list=[]
        input_list.append(str(letters_count))
        input_list.append(str(max_words))
        input_list.append(list_to_table(letters_list,"Letter"))
        input_list.append(list_to_table(vocab,"Word"))
        input_list.append(given_word)
        prompt=generate_prompt(
            input_list,
            prompt_template
        )
        return prompt,input_list
    
    
    def __func_validate(gpt_response,
                        prompt=None):
        try:
            if __func_clean_up(gpt_response)!=False:
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def __func_clean_up(gpt_response,
                        vocab=vocab,
                        prompt=None,):
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            num_list=gpt_response.get("num_list",[])
            # if len(gpt_response)>10:
            #     return False
            if num_list:
                for num in num_list:
                    if num >len(vocab):
                        return False
            word_list=[]
            if num_list:
                for num in num_list:
                    word_list.append(vocab[num-1])
            gpt_response["word_list"]=word_list
            return gpt_response            
        except:
            return False

    def get_fail_safe():
        fs = "Error:The output of GPT is illegal."
        return fs
    
    fail_safe=get_fail_safe()
    
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }
    prompt,input_list=create_prompt(letters_count,
                        max_words,
                        letters_list,
                        vocab,
                        given_word)
    
    output= safe_generate_response(
        prompt, 
        gpt_param, 
        10, 
        fail_safe, 
        __func_validate, 
        __func_clean_up
    )
    
    if verbose:
        print_run_prompts(
            prompt_template=prompt_template,
            player_id=player_id,
            prompt=prompt,
            output=output,
        )
    
    return output

def run_gpt_prompt_listener_decide(letters_count,
                              letters_list,
                              vocab,
                              given_word,
                              semantic_features,
                              player_id,
                              model,
                              verbose=True):
    prompt_template="prompt/listener_decide.txt"
    def create_prompt(letters_count,
                      letters_list,
                      vocab,
                      given_word,
                      semantic_features,
                      prompt_template=prompt_template):
        
        input_list=[]
        input_list.append(str(letters_count))
        input_list.append(list_to_table(letters_list,"Letter"))
        input_list.append(dict_list_to_str(vocab))
        input_list.append(given_word)
        input_list.append(json.dumps(semantic_features,indent=4))
        prompt=generate_prompt(
            input_list,
            prompt_template
        )
        return prompt,input_list
    
    
    def __func_validate(gpt_response,
                        prompt=None):
        try:
            if __func_clean_up(gpt_response):
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def __func_clean_up(gpt_response,
                        prompt=None,):
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            option=gpt_response.get("option","")
            if not option or not isinstance(option, str):
                return False
            option=option.strip().upper()
            options="ABCDEFGHIJKLMNOPQIST"
            if option in options[:len(semantic_features)]:
                return gpt_response  
            else:
                return False       
        except:
            return False

    def get_fail_safe():
        fs = "Error:The output of GPT is illegal."
        return fs
    
    fail_safe=get_fail_safe()
    
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }
    prompt,input_list=create_prompt(letters_count,
                        letters_list,
                        vocab,
                        given_word,
                        semantic_features)
    
    output= safe_generate_response(
        prompt, 
        gpt_param, 
        5, 
        fail_safe, 
        __func_validate, 
        __func_clean_up
    )
    
    if verbose:
        print_run_prompts(
            prompt_template=prompt_template,
            player_id=player_id,
            prompt=prompt,
            output=output,
        )
    
    return output

def run_gpt_prompt_select_feature(word,
                              verbose=False):
    prompt_template="prompt/select_features.txt"
    def create_prompt(word,prompt_template=prompt_template):
        
        input_list=[]
        input_list.append(word)
        prompt=generate_prompt(
            input_list,
            prompt_template
        )
        return prompt,input_list
    
    
    def __func_validate(gpt_response,
                        prompt=None):
        try:
            if __func_clean_up(gpt_response):
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def __func_clean_up(gpt_response,
                        prompt=None,):
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            all_catsegories = [
                'encyclopaedic',
                'function',
                'smell',
                'sound',
                'tactile',
                'taste',
                'taxonomic',
                'visual_colour',
                'visual_form_and_surface',
                'visual_motion'
            ]
            if set(gpt_response).issubset(all_catsegories):
                return gpt_response
            else:
                return False
        except:
            return False

    def get_fail_safe():
        fs = "Error:The output of GPT is illegal."
        return fs
    
    fail_safe=get_fail_safe()
    
    gpt_param = {
        "model":"gpt-4o-mini",
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }
    prompt,input_list=create_prompt(word)
    
    output= safe_generate_response(
        prompt, 
        gpt_param, 
        5, 
        fail_safe, 
        __func_validate, 
        __func_clean_up
    )
    
    if verbose:
        print_run_prompts(
            prompt_template=prompt_template,
            prompt=prompt,
            output=output,
        )
    
    return output

def run_gpt_prompt_select_synonyms(
                              object_features,
                              features_list:list,
                              model,
                              verbose=True):
    prompt_template="prompt/select_synonyms.txt"
    def create_prompt(
                      object_features:dict,
                      features_list:list,
                      prompt_template=prompt_template):
        
        input_list=[]
        input_list.append(str(object_features))
        all_features_str=""
        for i in range(len(features_list)):
            all_features_str+=f"{i}:"+str(features_list[i])+"\n"
        input_list.append(all_features_str)    
        
        prompt=generate_prompt(
            input_list,
            prompt_template
        )
        return prompt,input_list
    
    
    def __func_validate(gpt_response,
                        prompt=None):
        try:
            if __func_clean_up(gpt_response):
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def __func_clean_up(gpt_response,
                        prompt=None):
        try:
            gpt_response=json.loads(deal_json_format(gpt_response))
            num_list=gpt_response.get("num_list",[])
            if type(num_list) is list:
                for num in num_list:
                    if not isinstance(num,int) or num>=len(features_list):
                        return False
                return gpt_response
            else:
                return False
                    
        except:
            return False

    def get_fail_safe():
        fs = "Error:The output of GPT is illegal."
        return fs
    
    fail_safe=get_fail_safe()
    
    gpt_param = {
        "model":model,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0.1
    }
    prompt,input_list=create_prompt(object_features=object_features,
                                    features_list=features_list)
    
    output = safe_generate_response(
        prompt, 
        gpt_param, 
        10, 
        fail_safe, 
        __func_validate, 
        __func_clean_up
    )
    if verbose:
        print_run_prompts(
            prompt_template=prompt_template,
            player_id=None,
            prompt=prompt,
            output=output,
        )
    
    return output
