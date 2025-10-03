import pandas as pd
import os
import numpy as np
from llm_methods.gpt_structure import text_embedding_request
from llm_methods.run_gpt_prompt import run_gpt_prompt_select_feature
import json
import random

def find_most_similar_word(target_word, candidate_words,k=1):
        target = np.array(target_word).reshape(1, -1)  
        candidates = np.array(candidate_words)         
        
        target_norm = target / np.linalg.norm(target, axis=1, keepdims=True)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        
        similarities = np.dot(candidates_norm, target_norm.T).flatten()
        if len(similarities)>=k:
            top_k_indices = np.argsort(similarities)[-k:][::-1]
        else:
            top_k_indices = np.argsort(similarities)[::-1]

        top_k_results = [
            (candidates[i], similarities[i]) 
            for i in top_k_indices
        ]
        
        return top_k_indices.tolist(),top_k_results
    

class BRM_loader:
    def __init__(self):
        self.data_dir = './data/McRae-BRM-InPress'
        self.df= pd.read_excel(os.path.join(self.data_dir, 'FEATS_brm.xlsx'))
        self.df_list=dict()
        self.data_groupby()
        self.text_embedding=dict()
        
    def data_groupby(self):
        if self.df_list=={}:
            self.df = self.df.dropna()
            self.df['Feature'] = self.df['Feature'].str.replace('_', ' ')
            self.df['BR_Label'] = self.df['BR_Label'].str.replace('-', '_')
            self.df['Feature'] = self.df['Feature'].str.replace(r'^beh - ', 'living behavior: ', regex=True)
            self.df['Feature'] = self.df['Feature'].str.replace(r'^inbeh - ', 'non-living behavior: ', regex=True)
            grouped = self.df.groupby('BR_Label')
            for group_name, group in grouped:
                self.df_list[group_name] = group["Feature"].tolist()
        else:
            print("The data has been grouped by BR_Label.")
    
    def data_embedding(self):
        if self.df_list!={}:
            pass
        else:
            self.data_groupby()
        text_embedding=dict()
        for tp,feature in self.df_list.items():
            embedding_list=text_embedding_request(feature)
            text_embedding[tp]=np.array(embedding_list)
        self.text_embedding=text_embedding
        
    def save_embedding(self,name):
        if self.text_embedding!={}:
            pass
        else:
            self.data_embedding()
        os.makedirs(f"./data/{name}",exist_ok=True)
        
        for key ,value in self.text_embedding.items():
            np.save(f'./data/{name}/embeddings_{key}.npy', value)
        
    def load_embedding(self,name):
        folder_path=os.path.join("./data", name)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.npy'):
                    key = "_".join(file_name.split('_')[1:]).split('.')[0]
                    value = np.load(os.path.join(folder_path, file_name))
                    self.text_embedding[key] = value    
        else:
            self.data_embedding()
            
    def get_text_embedding(self):
        if self.text_embedding=={}:
            self.load_embedding()
        return self.text_embedding
    
    def get_df_list(self):
        if self.df_list=={}:
            self.data_groupby()
        return self.df_list
    
    def generate_features(self,word):
        features=dict()
        text_embedding=text_embedding_request(word)
        feature_select=run_gpt_prompt_select_feature(word)
        for tp,embeddings in self.text_embedding.items():
            if tp in feature_select:
                top_k_id,_=find_most_similar_word(text_embedding, embeddings,k=1)
                features[tp]=self.df_list[tp][top_k_id[0]]
            else:
                features[tp]=None
        return features
def load_object_feature_pairs(seed=None,count=None):
    pairs=dict()
    path="./data/McRae-BRM-InPress/CONCS_FEATS_concstats_brm.xlsx"
    df=pd.read_excel(path)
    # df=df.dropna()
    df['Feature'] = df['Feature'].str.replace('_', ' ')
    df['BR_Label'] = df['BR_Label'].str.replace('-', '_')
    df['Feature'] = df['Feature'].str.replace(r'^beh - ', 'living behavior: ', regex=True)
    df['Feature'] = df['Feature'].str.replace(r'^inbeh - ', 'non-living behavior: ', regex=True)
    groups=df.groupby("Concept")
    for concept,group in groups:
        pairs[concept]={
            'encyclopaedic':[],
            'function':[],
            'smell':[],
            'sound':[],
            'tactile':[],
            'taste':[],
            'taxonomic':[],
            'visual_colour':[],
            'visual_form_and_surface':[],
            'visual_motion':[]
            }
        for index,row in group.iterrows():
            pairs[concept][row['BR_Label']].append(row["Feature"])
    if seed and count:
        random_pairs=dict()
        random.seed(seed)
        keys=random.sample(list(pairs.keys()),k=count)
        for key in keys:
            random_pairs[key]=pairs[key]
        pairs=random_pairs
        random.seed(None)
    return pairs   

if __name__ == "__main__":
    pass
    
    