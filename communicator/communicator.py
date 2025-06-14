from communicator.WordDatabase import WordDatabase
from llm_methods.gpt_structure import text_embedding_request
from llm_methods.run_gpt_prompt import *
import random

class communicator:
    def __init__(self,letter_list,id):
        self.letter_list=letter_list
        self.word_database=WordDatabase()
        self.player_id=id
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
                player_id=self.player_id,
                failed_records=failed_records    
            )["word"]
        else:
            length=random.randint(2,6)
            word="".join(random.choices(self.letter_list,k=length))
        return word
    
    def listener_select(self,word,sf_dict:dict):
        word_exists=False
        try:
            word_nums=self.word_database.word_to_key_dict[word]
            weights=self.word_database.weight_output(word_nums,"listener")
            if weights==None:
                raise KeyError
            for i in reversed(range(len(weights))):
                if weights[i]<=0.1:
                    self.word_database.delete(word_nums[i])
                    word_nums.pop(i)
            weights=self.word_database.weight_output(word_nums,"listener")
            num=random.choices(word_nums,weights=weights,k=1)[0]
            # num=random.choice(word_nums)
            semantic_features=self.word_database.word_dict[num].toFeatures()
            for key,value in sf_dict.items():
                if value==semantic_features:
                    word_exists=True
                    self.word_database.word_dict[num].listen_fail_count-=1
                    return word_exists,key,num
        except KeyError as e:
            pass
        except Exception as e:
            print("communicator: lister_select error:")
            print(e)
        if self.word_database.get_word_list():
            # resembling_list=run_gpt_prompt_select_resembling_words(
            #     letters_count=len(self.letter_list),
            #     max_words=10,
            #     letters_list=self.letter_list,
            #     vocab=self.word_database.get_word_list(),
            #     given_word=word,
            #     player_id=self.player_id
            # )
            resembling_list=self.word_database.search_resembling_word(word)
            if resembling_list:
                random.shuffle(resembling_list)
        else:
            resembling_list=[]
        word_num_list=[]
        if resembling_list:
            for resem_word in resembling_list:
                word_num_list+=self.word_database.word_to_key_dict[resem_word]
            vocab_list=[self.word_database.word_dict[num].todict_wo_object() for num in word_num_list]
            choice=run_gpt_prompt_listener_decide(
                len(self.letter_list),
                self.letter_list,
                vocab_list,
                word,
                sf_dict,
                player_id=self.player_id
            )["option"]
        else:
            select_list=["A","B","C","D","E"]
            choice=random.choice(select_list)
        return word_exists,choice,None
        
        