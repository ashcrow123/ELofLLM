from data_loader.BRM_loader import BRM_loader
import numpy as np
import json
import os

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
        self.fail_count=0
    def todict(self):
        return {
            "obj": self.obj,
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
        self.embeddings_list=[]
        self.search_dict={
            tp: {feature:[] for feature in features_loader.df_list[tp]} for tp in list(features_loader.df_list.keys())    
        }
        self.word_dict=dict()
        self.word_to_key_dict=dict()
    def add_word(self,
                text_embedding,
                word: str,
                obj: str,
                encyclopaedic: str,
                function: str,
                smell: str,
                sound: str,
                tactile: str,
                taste: str,
                taxonomic: str,
                visual_colour: str,
                visual_form_and_surface: str,
                visual_motion: str):
        
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
        
        length=str(len(self.word_dict))
        self.word_dict[str(length)]=new_word
        try:
            self.word_to_key_dict[word].append(length)
        except:
            self.word_to_key_dict[word]=[length]
        for tp in self.search_dict.keys():
            word_features=new_word.todict()[tp]
            if not word_features:
                continue
            for feature in word_features:
                try:
                    self.search_dict[tp][feature].append(length) 
                except:
                    self.search_dict[tp][feature]=[length]
        self.embeddings_list.append(text_embedding)
        return
    
    def change_word(self,
                encyclopaedic: str,
                function: str,
                smell: str,
                sound: str,
                tactile: str,
                taste: str,
                taxonomic: str,
                visual_colour: str,
                visual_form_and_surface: str,
                visual_motion: str,
                origin_word:str,
                replace_word:str):
        search_word_list=self.search_word(
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
        for key in search_word_list:
            if self.word_dict[key]==origin_word:
                self.word_dict[key].change_word(replace_word)
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
        lists=[]
        for tp,features in search_dict.items():
            if features==[]:
                continue
            for feature in features:
                try:
                    lists.append(self.search_dict[tp][feature]) if self.search_dict[tp][feature]!=[] else None
                except:
                    pass
        intersection=get_Intersection(lists)
        for elem in intersection:
            elem_dict=self.word_dict[elem].todict()
            for key in list(search_dict.keys()):
                if elem_dict[key]==search_dict[key]:
                    pass
                else:
                    intersection.remove(elem)
                    break
        if len(intersection)==0:
            return []
        else:
            return intersection
        
    def search_near_synonyms(self,text_embedding):
        top_k_ids,_=find_most_similar_word(text_embedding,self.embeddings_list,k=10,similarity_threshold=0.85)
        syn_list=[]
        for id in top_k_ids:
            word=self.word_dict[str(id)]
            syn_list.append(word)
        return syn_list
    #TODO 修改使得匹配新的数据结构        
    def search_resembling_word(self,target_word):
        try:
            target_blocks = split_cv_blocks(target_word)
        except ValueError as e:
            print(f"Invalid target word: {e}")
            return []

        distances = []
        for word in self.word_dict.keys():
            try:
                blocks = split_cv_blocks(word)
                dist = edit_distance_cv_blocks(target_blocks, blocks)
                distances.append((word, dist))
            except ValueError:
                continue  # 跳过非法词

        distances.sort(key=lambda x: x[1])
        return distances[:10] if len(distances) >= 10 else distances
         
    #TODO how to save &load word_dict   
    def load(self,path):
        with open(os.path.join(path,"word_dict.json"),"r",encoding="utf-8") as f:
            load_dict=json.load(f)
        for key in load_dict.keys():
            self.word_dict[key]=Word(**load_dict[key])
        with open(os.path.join(path,"word_to_key_dict.json"),"r",encoding="utf-8") as f:
            self.word_to_key_dict=json.load(f)
        with open(os.path.join(path,"search_dict.json"),"r",encoding="utf-8") as f:
            self.search_dict=json.load(f)
        with open(os.path.join(path,"embeddings_list.json"),"r",encoding="utf-8") as f:
            self.embeddings_list=json.load(f)
        return
    
    def save(self,path):
        os.makedirs(path,exist_ok=True)
        save_dict=dict()
        for key in self.word_dict.keys():
            save_dict[key]=self.word_dict[key].todict()
        with open(os.path.join(path,"word_dict.json"),"w",encoding="utf-8") as f:
            json.dump(save_dict,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"embeddings_list.json"),"w",encoding="utf-8") as f:
            json.dump(self.embeddings_list,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"word_to_key_dict.json"),"w",encoding="utf-8") as f:
            json.dump(self.word_to_key_dict,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"search_dict.json"),"w",encoding="utf-8") as f:
            json.dump(self.search_dict,f,ensure_ascii=False,indent=4)
        return
       
    def get_word_list(self):
        word_list=list(self.word_to_key_dict.keys())
        return word_list