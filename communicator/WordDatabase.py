from data_loader.BRM_loader import BRM_loader
from llm_methods.gpt_structure import text_embedding_request
import numpy as np
import json
import os
import warnings
import random
import time
def split_cv_blocks(word):
    """将单词分割成辅音+元音组合块"""
    blocks = []
    i = 0
    while i < len(word) - 1:
        c, v = word[i], word[i+1]
        if c.lower() in "bcdfghjklmnpqrstvwxyz" and v.lower() in "aeiou":
            blocks.append(c + v)
            i += 2
        else:
            raise ValueError(f"Invalid CV block at position {i}: {c+v}")
    if i < len(word):  # 遇到奇数长度或非法结尾
        raise ValueError(f"Word has leftover characters at the end: {word[i:]}")
    return blocks

def edit_distance_cv_blocks(blocks1, blocks2):
    """给定两个块序列，计算它们的编辑距离"""
    m, n = len(blocks1), len(blocks2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if blocks1[i - 1] == blocks2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]

def Jaccard_similarity(blocks_1, blocks_2, n):
    def n_grams(blocks, n):
        """生成n-gram"""
        return set(tuple(blocks[i:i+n]) for i in range(len(blocks) - n + 1))
    set1=n_grams(blocks_1, n)
    set2=n_grams(blocks_2, n)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def get_Intersection(lists):
    # 计算多个列表的交集
    if not lists:
        return []
    intersection = set(lists[0])
    for lst in lists[1:]:
        intersection.intersection_update(lst)
    return list(intersection)

def find_most_similar_word(target_word, candidate_words, k=1, similarity_threshold=0.0):
    # 转换 target 为二维向量 (1, n)
    target = np.array(target_word).reshape(1, -1)

    # 检查 candidates 是否为空
    if len(candidate_words) == 0:
        return [], []

    candidates = np.array(candidate_words)

    # 如果 candidates 是一维（只有一个向量），转换为二维 (1, n)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)

    # 防止除以 0 的错误（向量全为 0 时范数为 0）
    def safe_normalize(x, axis):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        norm[norm == 0] = 1
        return x / norm

    target_norm = safe_normalize(target, axis=1)
    candidates_norm = safe_normalize(candidates, axis=1)

    # 计算余弦相似度
    similarities = np.dot(candidates_norm, target_norm.T).flatten()

    # 获取 top-k 的索引（按相似度降序）
    top_k = min(k, len(similarities))
    sorted_indices = np.argsort(similarities)[::-1]

    # 过滤相似度小于阈值的结果
    filtered = [
        (i, candidates[i], similarities[i])
        for i in sorted_indices[:top_k]
        if similarities[i] >= similarity_threshold
    ]

    top_k_indices = [i for i, _, _ in filtered]
    top_k_results = [(vec, sim) for _, vec, sim in filtered]
    
    return top_k_indices, top_k_results

class Word:
    def __init__(self,
                 obj:str, 
                 word: str,
                 speak_fail_count=0,
                 listen_fail_count=0,
                 encyclopaedic=[],
                 function=[],
                 smell=[],
                 sound=[],
                 tactile=[],
                 taste=[],
                 taxonomic=[],
                 visual_colour=[],
                 visual_form_and_surface=[],
                 visual_motion=[]):
        
        self.obj = obj
        self.word = word
        self.encyclopaedic = encyclopaedic
        self.function = function
        self.smell = smell
        self.sound = sound
        self.tactile = tactile
        self.taste = taste
        self.taxonomic = taxonomic
        self.visual_colour = visual_colour
        self.visual_form_and_surface = visual_form_and_surface
        self.visual_motion = visual_motion
        self.speak_fail_count=speak_fail_count
        self.listen_fail_count=listen_fail_count
    def todict(self):
        return {
            "obj": self.obj,
            "word": self.word,
            "speak_fail_count":self.speak_fail_count,
            "listen_fail_count":self.listen_fail_count,
            "encyclopaedic": self.encyclopaedic,
            "function": self.function,
            "smell": self.smell,
            "sound": self.sound,
            "tactile": self.tactile,
            "taste": self.taste,
            "taxonomic": self.taxonomic,
            "visual_colour": self.visual_colour,
            "visual_form_and_surface": self.visual_form_and_surface,
            "visual_motion": self.visual_motion
        }
    def todict_wo_object(self):
        return {
            "word": self.word,
            "encyclopaedic": self.encyclopaedic,
            "function": self.function,
            "smell": self.smell,
            "sound": self.sound,
            "tactile": self.tactile,
            "taste": self.taste,
            "taxonomic": self.taxonomic,
            "visual_colour": self.visual_colour,
            "visual_form_and_surface": self.visual_form_and_surface,
            "visual_motion": self.visual_motion
        }
    def toFeatures(self):
        return {
            "encyclopaedic": self.encyclopaedic,
            "function": self.function,
            "smell": self.smell,
            "sound": self.sound,
            "tactile": self.tactile,
            "taste": self.taste,
            "taxonomic": self.taxonomic,
            "visual_colour": self.visual_colour,
            "visual_form_and_surface": self.visual_form_and_surface,
            "visual_motion": self.visual_motion
        }
    def change_word(self,word):
        self.word=word
        return
    
class WordDatabase:
    def __init__(self,
                 features_name="BRM",
                ):
        self.features_name = features_name
        features_loader=BRM_loader()
        features_loader.load_embedding(name=self.features_name)
        self.embeddings_list=dict()
        # self.search_dict={
        #     tp: {feature:[] for feature in features_loader.df_list[tp]} for tp in list(features_loader.df_list.keys())    
        # }
        # self.search_dict={
        #     tp: dict() for tp in list(features_loader.df_list.keys())    
        # }
        self.word_dict=dict()
        self.word_to_key_dict=dict()
        self.obj_dict=dict()
    def add_word(self,
                text_embedding,
                word: str,
                obj: str,
                encyclopaedic: list,
                function: list,
                smell: list,
                sound: list,
                tactile: list,
                taste: list,
                taxonomic: list,
                visual_colour: list,
                visual_form_and_surface: list,
                visual_motion: list):
        if word in self.get_word_list():
            word_num_list=self.word_to_key_dict[word]
            for num in word_num_list:
                features=self.word_dict[num].toFeatures()
                if features=={
                    "encyclopaedic": encyclopaedic,
                    "function": function,
                    "smell": smell,
                    "sound": sound,
                    "tactile": tactile,
                    "taste": taste,
                    "taxonomic": taxonomic,
                    "visual_colour": visual_colour,
                    "visual_form_and_surface": visual_form_and_surface,
                    "visual_motion": visual_motion
                }:
                    warnings.warn(f"The word {word} has already in the word database.",RuntimeWarning)
                    return
                    
        new_word=Word(
            obj=obj,
            word=word,
            encyclopaedic=encyclopaedic,
            function=function,
            smell=smell,
            sound=sound,
            tactile=tactile,
            taste=taste,
            taxonomic=taxonomic,
            visual_colour=visual_colour,
            visual_form_and_surface=visual_form_and_surface,
            visual_motion=visual_motion,
        )
        
        new_num=str(np.max([int(num) for num in self.word_dict.keys()])+1) if self.word_dict else "0"
        self.word_dict[str(new_num)]=new_word
        try:
            self.word_to_key_dict[word].append(new_num)
        except:
            self.word_to_key_dict[word]=[new_num]
        # for tp in self.search_dict.keys():
        #     word_features=new_word.todict()[tp]
        #     if not word_features:
        #         continue
        #     for feature in word_features:
        #         try:
        #             self.search_dict[tp][feature].append(new_num) 
        #         except:
        #             self.search_dict[tp][feature]=[new_num]
        try:
            self.obj_dict[obj].append(new_num)
        except:
            self.obj_dict[obj]=[new_num]
        
        if not (obj in list(self.embeddings_list.keys())):
            self.embeddings_list[obj]=text_embedding
          
        return   
    def search_word(self, 
                    encyclopaedic,
                    function,
                    smell,
                    sound,
                    tactile,
                    taste,
                    taxonomic,
                    visual_colour,
                    visual_form_and_surface,
                    visual_motion,):
        search_dict={
            "encyclopaedic": encyclopaedic,
            "function": function,
            "smell": smell,
            "sound": sound,
            "tactile": tactile,
            "taste": taste,
            "taxonomic": taxonomic,
            "visual_colour": visual_colour,
            "visual_form_and_surface": visual_form_and_surface,
            "visual_motion": visual_motion,
        }
        # lists=[]
        # for tp,features in search_dict.items():
        #     if features==[]:
        #         continue
        #     for feature in features:
        #         try:
        #             lists.append(self.search_dict[tp][feature]) if self.search_dict[tp][feature]!=[] else None
        #         except:
        #             pass
        # intersection=get_Intersection(lists)
        # for num in intersection:
        #     features_dict=self.word_dict[num].toFeatures()
        #     if features_dict!=search_dict:
        #         intersection.remove(num)
        # return intersection
        num_list=[]
        for num,word in self.word_dict.items():
            word_features=word.toFeatures()
            if word_features==search_dict:
                num_list.append(num)
        return num_list
        
    def search_near_synonyms(self,obj,text_embedding):
        top_k_ids,_=find_most_similar_word(text_embedding,list(self.embeddings_list.values()),k=10,similarity_threshold=0.84)
        syn_list=[]
        all_obj=list(self.embeddings_list.keys())
        top_k_obj=[all_obj[id] for id in top_k_ids]
        for i in top_k_obj:
            words=self.obj_dict[i]
            syn_list+=words
        try:
            num_list=self.obj_dict[obj]
            for num in num_list:
                syn_list.remove(num)
        except:
            pass
        return syn_list
    #TODO 修改使得匹配新的数据结构        
    def search_resembling_word(self,target_word):
        try:
            target_blocks = split_cv_blocks(target_word)
        except ValueError as e:
            print(f"Invalid target word: {e}")
            return []

        distances = []
        for word in self.word_to_key_dict.keys():
            if word!=target_word:
                try:
                    blocks = split_cv_blocks(word)
                    # dist = edit_distance_cv_blocks(target_blocks, blocks)
                    # similarity=1-dist/max(len(blocks),len(target_blocks))
                    similarity= Jaccard_similarity(target_blocks, blocks, n=2)+ Jaccard_similarity(target_blocks, blocks, n=3)+\
                                Jaccard_similarity(target_blocks, blocks, n=4)+Jaccard_similarity(target_blocks, blocks, n=5)
                    # similarity=1/(1+dist)
                    if similarity>0:
                        distances.append(word)
                except Exception as e:
                    continue  # 跳过非法词
        random.shuffle(distances)
        return distances
    def weight_output(self,num_list,identity,beta=1.3):
        if not (identity in ["speaker","listener"]):
            raise ValueError("identity must be 'speaker' or 'listener'. ")
        if not num_list:
            return None
        if not set(num_list).issubset(set(self.word_dict.keys())):
            raise ValueError("There are numbers in the list that are not in the word database.")
        total=0
        weights=[]
        for num in num_list:
            if identity=="speaker":
                word_fail_count=self.word_dict[num].speak_fail_count
            else:
                word_fail_count=self.word_dict[num].listen_fail_count
            exp_num=np.exp(-word_fail_count*beta)
            total+=exp_num
            weights.append(exp_num)
        weights=[weight/total for weight in weights]
        return weights     
    #TODO how to save &load word_dict   
    def load(self,path):
        with open(os.path.join(path,"word_dict.json"),"r",encoding="utf-8") as f:
            load_dict=json.load(f)
        for key in load_dict.keys():
            self.word_dict[key]=Word(**load_dict[key])
        with open(os.path.join(path,"word_to_key_dict.json"),"r",encoding="utf-8") as f:
            self.word_to_key_dict=json.load(f)
        # with open(os.path.join(path,"embeddings_list.json"),"r",encoding="utf-8") as f:
        #     self.embeddings_list=json.load(f)
        with open(os.path.join(path,"obj_dict.json"),"r",encoding="utf-8") as f:
            self.obj_dict=json.load(f)
        embeddings_list=text_embedding_request(list(self.obj_dict.keys()))
        for i in range(len(embeddings_list)):
            self.embeddings_list[list(self.obj_dict.keys())[i]]=embeddings_list[i]
        time.sleep(2)
        return
    
    def save(self,path):
        os.makedirs(path,exist_ok=True)
        save_dict=dict()
        for key in self.word_dict.keys():
            save_dict[key]=self.word_dict[key].todict()
        with open(os.path.join(path,"word_dict.json"),"w",encoding="utf-8") as f:
            json.dump(save_dict,f,ensure_ascii=False,indent=4)
        # with open(os.path.join(path,"embeddings_list.json"),"w",encoding="utf-8") as f:
        #     json.dump(self.embeddings_list,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"word_to_key_dict.json"),"w",encoding="utf-8") as f:
            json.dump(self.word_to_key_dict,f,ensure_ascii=False,indent=4)
        # with open(os.path.join(path,"search_dict.json"),"w",encoding="utf-8") as f:
        #     json.dump(self.search_dict,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"obj_dict.json"),"w",encoding="utf-8") as f:
            json.dump(self.obj_dict,f,ensure_ascii=False,indent=4)
        return
       
    def get_word_list(self):
        word_list=list(self.word_to_key_dict.keys())
        return word_list
    
    def delete(self,num):
        num=str(num)
        if not (num in list(self.word_dict.keys())):
            warnings.warn(f"The num {num} is not in the word database.",RuntimeWarning)
            return
        del self.word_dict[num]
        word_remove_list=[]
        for word,num_list in self.word_to_key_dict.items():
            if num in num_list:
                num_list.remove(num)
                if num_list==[]:
                    word_remove_list.append(word)
        for word in word_remove_list:
            del self.word_to_key_dict[word]
        
        obj_remove_list=[]            
        for obj,num_list in self.obj_dict.items():
            if num in num_list:
                num_list.remove(num)
                if num_list==[]:
                    obj_remove_list.append(obj)
        for obj in obj_remove_list:
            del self.obj_dict[obj]
            del self.embeddings_list[obj]
        
        # for tp in self.search_dict.keys():
        #     for feature in tp.keys():
        #         if num in self.search_dict[tp][feature]:
        #             self.search_dict[tp][feature].remove(num)
        
        