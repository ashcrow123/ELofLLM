from communicator.WordDatabase import WordDatabase
from llm_methods.gpt_structure import text_embedding_request
from llm_methods.run_gpt_prompt import *
import random

class communicator:
    def __init__(self,letter_list,num):
        self.letter_list=letter_list
        self.word_database=WordDatabase()
        self.player_num=num
    def save(self,path):
        self.word_database.save(path)
    def load(self,path):
        self.word_database.load(path)  
          
    def generate_new_word(
        self,                   
        vocab,
        obj_features,
        failed_records,
    ):
        # if vocab or failed_records:
        if vocab:
            word=run_gpt_prompt_speaker_generate(
                len(self.letter_list),
                self.letter_list,
                vocab,
                obj_features,
                player_num=self.player_num,
                failed_records=failed_records    
            )
        else:
            length=random.randint(3,12)
            word="".join(random.choices(self.letter_list,k=length))
        return word
    
    def listener_select(self,word,sf_dict):
        if self.word_database.get_word_list():
            resembling_list=run_gpt_prompt_select_resembling_words(
                letters_count=len(self.letter_list),
                max_words=10,
                letters_list=self.letter_list,
                vocab=self.word_database.get_word_list(),
                given_word=word,
                player_num=self.player_num
            )
        else:
            resembling_list=[]
        word_num_list=[]
        if resembling_list:
            for word in resembling_list:
                word_num_list+=self.word_database.word_to_key_dict[word]
            vocab_list=[self.word_database.word_dict[num].todict_wo_object() for num in word_num_list]
            choice=run_gpt_prompt_listener_decide(
                len(self.letter_list),
                self.letter_list,
                vocab_list,
                word,
                sf_dict,
                player_num=self.player_num
            )
        else:
            select_list=["A","B","C"]
            choice=random.choices(select_list,k=1)[0]
        return choice
        
        