from data_loader.BRM_loader import BRM_loader
from llm_methods.gpt_structure import text_embedding_request
import numpy as np
import json
import os
import warnings
import random
import time
def split_cv_blocks(word):
    """Split words into consonant + vowel chunks"""
    blocks=word.split("-")
    return blocks

def edit_distance_cv_blocks(blocks1, blocks2):
    """Given two block sequences, calculate their edit distance"""
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
        """generate n-gram"""
        return set(tuple(blocks[i:i+n]) for i in range(len(blocks) - n + 1))
    set1=n_grams(blocks_1, n)
    set2=n_grams(blocks_2, n)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def get_Intersection(lists):
    # Compute the intersection of multiple lists
    if not lists:
        return []
    intersection = set(lists[0])
    for lst in lists[1:]:
        intersection.intersection_update(lst)
    return list(intersection)


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
                 model,
                ):
        self.word_dict=dict()
        self.word_to_key_dict=dict()
        self.obj_dict=dict()
        with open(f"data/{model}_network.json","r") as f:
            self.synonyms_search_dict=json.load(f)
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
        try:
            self.obj_dict[obj].append(new_num)
        except:
            self.obj_dict[obj]=[new_num]
        
          
        return
    def change_word(self,num,word):
        num=str(num)
        old_word=self.word_dict[num].word
        if old_word==word:
            return
        self.word_dict[num].change_word(word)
        self.word_to_key_dict[old_word].remove(num)
        if word in self.word_to_key_dict:
            self.word_to_key_dict[word].append(num)
        else:
            self.word_to_key_dict[word]=[num]
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
        num_list=[]
        for num,word in self.word_dict.items():
            word_features=word.toFeatures()
            if word_features==search_dict:
                num_list.append(num)
        return num_list
        
    def search_near_synonyms(self,obj):
        synonyms_list=self.synonyms_search_dict[obj]
        num_list=[]
        for syn_obj in synonyms_list:
            if syn_obj in self.obj_dict:
                num_list+=self.obj_dict[syn_obj]
        if obj in self.obj_dict:
            self_list=self.obj_dict[obj]
            for num in self_list:
                if num in num_list:
                    num_list.remove(num)
        return num_list       
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
                    similarity= Jaccard_similarity(target_blocks, blocks, n=2)+ Jaccard_similarity(target_blocks, blocks, n=3)+\
                                Jaccard_similarity(target_blocks, blocks, n=4)+Jaccard_similarity(target_blocks, blocks, n=5)
                    if similarity>0:
                        distances.append(word)
                except Exception as e:
                    continue  
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
     
    def load(self,path):
        with open(os.path.join(path,"word_dict.json"),"r",encoding="utf-8") as f:
            load_dict=json.load(f)
        for key in load_dict.keys():
            self.word_dict[key]=Word(**load_dict[key])
        with open(os.path.join(path,"word_to_key_dict.json"),"r",encoding="utf-8") as f:
            self.word_to_key_dict=json.load(f)
        with open(os.path.join(path,"obj_dict.json"),"r",encoding="utf-8") as f:
            self.obj_dict=json.load(f)
        return
    
    def save(self,path):
        os.makedirs(path,exist_ok=True)
        save_dict=dict()
        for key in self.word_dict.keys():
            save_dict[key]=self.word_dict[key].todict()
        with open(os.path.join(path,"word_dict.json"),"w",encoding="utf-8") as f:
            json.dump(save_dict,f,ensure_ascii=False,indent=4)
        with open(os.path.join(path,"word_to_key_dict.json"),"w",encoding="utf-8") as f:
            json.dump(self.word_to_key_dict,f,ensure_ascii=False,indent=4)
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
            
        
