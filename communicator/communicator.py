from communicator.WordDatabase import WordDatabase
from llm_methods.gpt_structure import text_embedding_request
from llm_methods.run_gpt_prompt import *
import random

class communicator:
    def __init__(self,letter_list,id,max_length,model):
        self.letter_list=letter_list
        self.word_database=WordDatabase(model=model)
        self.player_id=id
        self.max_length=max_length
        self.model=model
    def save(self,path):
        self.word_database.save(path)
    def load(self,path):
        self.word_database.load(path)  
        print(f"player_{self.player_id} has loaded word database.")  
    def generate_new_word(
        self,                   
        vocab,
        obj_features,
        failed_records,
    ):
        if vocab:
            word=run_gpt_prompt_speaker_generate(
                len(self.letter_list),
                self.letter_list,
                vocab,
                obj_features,
                player_id=self.player_id,
                failed_records=failed_records,
                max_length=self.max_length,
                model=self.model    
            )["word"]
        else:
            length=random.randint(1,self.max_length)
            word="-".join(random.choices(self.letter_list,k=length))
        return word
    
    def listener_select(self,word,sf_dict:dict):
        word_exists=False
        try:
            word_nums=self.word_database.word_to_key_dict[word]
            choice_list=[]
            for num in word_nums:
                semantic_features=self.word_database.word_dict[num].toFeatures()
                for key,value in sf_dict.items():
                    if value==semantic_features:
                        choice_list.append((key,num))
            if choice_list:
                word_exists=True
                pair=random.choice(choice_list)
                return word_exists,pair[0],pair[1]
        except KeyError as e:
            pass
        except Exception as e:
            print("communicator: lister_select error:")
            print(e)
        if self.word_database.get_word_list():
            resembling_list=run_gpt_prompt_select_resembling_words(
                letters_count=len(self.letter_list),
                max_words=10,
                letters_list=self.letter_list,
                vocab=self.word_database.get_word_list(),
                given_word=word,
                player_id=self.player_id,
                model=self.model,
            )['word_list']
            # resembling_list=self.word_database.search_resembling_word(word)
            # resembling_list=list(self.word_database.word_to_key_dict.keys())
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
                player_id=self.player_id,
                model=self.model
            )["option"]
        else:
            select_list=["A","B","C","D","E"]
            choice=random.choice(select_list)
        return word_exists,choice,None
        
        