from communicator.communicator import communicator
from llm_methods.gpt_structure import text_embedding_request
from data_loader.BRM_loader import BRM_loader,load_object_feature_pairs
import random
import os
import json
def select_letters(num=10,seed=None):
    random.seed(seed)
    all_letters=[]
    for i in "bcdfghjklmnpqrstvwxyz":
        for j in "aeiou":
            all_letters.append(i+j)
    letters=random.sample(all_letters,k=num)
    random.seed(None)
    return letters

#TODO 设置数据集
class Referential_Game:
    def __init__(
        self,
        name:str,
        player_num:int,
        letter_list:list,
        comm_num,
        save_interval,
        obj_loader,
        max_length,
        model_list
    ):
        self.name=name
        self.comm_num=comm_num
        if player_num<2 or player_num%2!=0:
            raise ValueError("The number of players must be an even number and greater than 1.")
        os.makedirs(f"./sim_storage/{self.name}",exist_ok=True)
        self.features_loader=BRM_loader()
        self.features_loader.load_embedding("BRM")
        self.max_length=max_length
        self.player_num=player_num 
        self.letter_list=letter_list
        self.model_list=model_list
        self.players=dict()
        for i in range(player_num):
            self.players[str(i)]=communicator(self.letter_list,id=str(i),max_length=self.max_length,model=self.model_list[i])
        self.round=0
        self.obj_loader=obj_loader
        self.save_interval=save_interval
    def communicate(
                self,
                speaker_id:str,
                listener_id:str,
                obj_dict:dict):
        speaker=self.players[speaker_id]
        listener=self.players[listener_id]
        round=0
        success=False
        word_list=[]
        choices_list=[]
        corr_obj=list(self.obj_loader.keys())[(self.round-1)%len(self.obj_loader.keys())]
        corr_obj_embedding=text_embedding_request(corr_obj)
        obj_features_dict={**obj_dict,**{corr_obj:self.obj_loader[corr_obj]}}
        obj_pairs=[(key,value) for key,value in obj_features_dict.items()]
        random.shuffle(obj_pairs)
        obj_features_dict=dict(obj_pairs)
        obj_list=list(obj_features_dict.keys())
        select_features_dict=dict()
        choices_dict=dict()
        choices="ABCDE"
        for i in range(len(obj_list)):
            choices_dict[choices[i]]=obj_list[i]
            select_features_dict[choices[i]]=obj_features_dict[obj_list[i]]
        speaker_known_vocab_num=speaker.word_database.search_word(**obj_features_dict[corr_obj])
        random.shuffle(speaker_known_vocab_num)
        speaker_known_vocab=[speaker.word_database.word_dict[num].word for num in speaker_known_vocab_num]
        speaker_near_synonyms=[speaker.word_database.word_dict[num].todict_wo_object() for num in speaker.word_database.search_near_synonyms(corr_obj)]
        # speaker_near_synonyms=[speaker.word_database.word_dict[num].todict_wo_object() for num in speaker.word_database.word_dict]
        speaker_short_memory=[]
        speaker_used_num=speaker_known_vocab_num[0] if speaker_known_vocab_num else None
        # speaker_known_vocab_num=[]
        while True: 
            round+=1
            if round<=self.comm_num:
                if speaker_known_vocab_num:
                    word_num=speaker_known_vocab_num[0]
                    speaker_known_vocab_num.remove(word_num)
                    word=speaker.word_database.word_dict[word_num].word
                    word_exists,choice,used_num=listener.listener_select(
                    word=word,
                    sf_dict=select_features_dict,
                    )
                    choices_list.append(choice)
                    if choices_dict[choice]==corr_obj:
                        success=True
                        if not word_exists:
                            listener_word_list=listener.word_database.search_word(**obj_features_dict[corr_obj])
                            if listener_word_list:
                                listener_used_num=listener_word_list[0]
                                listener.word_database.change_word(listener_used_num,word)
                            else:
                                listener.word_database.add_word(
                                    text_embedding=corr_obj_embedding,
                                    word=word,
                                    obj=corr_obj,
                                    **obj_features_dict[corr_obj],
                                )
                            
                        word_list.append(word)
                        
                        break
                    else:
                        short_memory={
                            word:obj_features_dict[choices_dict[choice]],
                        }
                        speaker_short_memory.append(short_memory)
                        word_list.append(word)
                        listener_used_num=used_num
                else:
                    word=speaker.generate_new_word(
                        vocab=speaker_near_synonyms,
                        obj_features=obj_features_dict[corr_obj],
                        failed_records=speaker_short_memory,
                    )  
                    word_exists,choice,used_num=listener.listener_select(
                        word=word,
                        sf_dict=select_features_dict,
                    )
                    choices_list.append(choice)
                    if choices_dict[choice]==corr_obj:
                        success=True
                        if speaker_known_vocab:
                            speaker.word_database.change_word(speaker_used_num,word)
                        else:
                            speaker.word_database.add_word(
                            text_embedding=corr_obj_embedding,
                            word=word,
                            obj=corr_obj,
                            **obj_features_dict[corr_obj],
                        )
                        if not word_exists:
                            listener_word_list=listener.word_database.search_word(**obj_features_dict[corr_obj])
                            if listener_word_list:
                                listener_used_num=listener_word_list[0]
                                listener.word_database.change_word(listener_used_num,word)
                            else:
                                listener.word_database.add_word(
                                    text_embedding=corr_obj_embedding,
                                    word=word,
                                    obj=corr_obj,
                                    **obj_features_dict[corr_obj],
                                )
                            
                        word_list.append(word)
                        break
                    else:
                        short_memory={
                            word:obj_features_dict[choices_dict[choice]],
                        }
                        speaker_short_memory.append(short_memory)
                        word_list.append(word)
                        listener_used_num=used_num
            else:
                break
        return {
            "speaker_id":speaker_id,
            "listener_id":listener_id,
            "success":success,
            "word_list":word_list,
            "obj_list":obj_list,
            "corr_obj":corr_obj,
            "choices_list":choices_list,
            "speaker_near_synonyms":[word['word'] for word in speaker_near_synonyms],
            "speaker_known_vocab":speaker_known_vocab
        }
    def test_communicate(
                    self,
                    speaker_id:str,
                    listener_id:str,
                    obj_dict:dict):
            speaker=self.players[speaker_id]
            listener=self.players[listener_id]
            round=0
            success=False
            word_list=[]
            choices_list=[]
            corr_obj=list(self.obj_loader.keys())[(self.round-1)%len(self.obj_loader.keys())]
            corr_obj_embedding=text_embedding_request(corr_obj)
            obj_features_dict={**obj_dict,**{corr_obj:self.obj_loader[corr_obj]}}
            obj_pairs=[(key,value) for key,value in obj_features_dict.items()]
            random.shuffle(obj_pairs)
            obj_features_dict=dict(obj_pairs)
            obj_list=list(obj_features_dict.keys())
            select_features_dict=dict()
            choices_dict=dict()
            choices="ABCDE"
            for i in range(len(obj_list)):
                choices_dict[choices[i]]=obj_list[i]
                select_features_dict[choices[i]]=obj_features_dict[obj_list[i]]
            speaker_known_vocab_num=speaker.word_database.search_word(**obj_features_dict[corr_obj])
            random.shuffle(speaker_known_vocab_num)
            speaker_known_vocab=[speaker.word_database.word_dict[num].word for num in speaker_known_vocab_num]
            speaker_near_synonyms=[speaker.word_database.word_dict[num].todict_wo_object() for num in speaker.word_database.search_near_synonyms(corr_obj)]
            speaker_short_memory=[]
            speaker_known_vocab_num=[]
            while True: 
                round+=1
                if round<=self.comm_num:
                    if speaker_known_vocab_num:
                        word_num=speaker_known_vocab_num[0]
                        speaker_known_vocab_num.remove(word_num)
                        word=speaker.word_database.word_dict[word_num].word
                        word_exists,choice,used_num=listener.listener_select(
                        word=word,
                        sf_dict=select_features_dict,
                        )
                        choices_list.append(choice)
                        if choices_dict[choice]==corr_obj:
                            success=True
                            word_list.append(word)
                            break
                        else:
                            short_memory={
                                word:obj_features_dict[choices_dict[choice]],
                            }
                            speaker_short_memory.append(short_memory)
                            word_list.append(word)
                    else:
                        word=speaker.generate_new_word(
                            vocab=speaker_near_synonyms,
                            obj_features=obj_features_dict[corr_obj],
                            failed_records=speaker_short_memory,
                        )  
                        word_exists,choice,used_num=listener.listener_select(
                            word=word,
                            sf_dict=select_features_dict,
                        )
                        choices_list.append(choice)
                        if choices_dict[choice]==corr_obj:
                            success=True
                            word_list.append(word)
                            break
                        else:
                            short_memory={
                                word:obj_features_dict[choices_dict[choice]],
                            }
                            speaker_short_memory.append(short_memory)
                            word_list.append(word)
                else:
                    break
            return {
                "speaker_id":speaker_id,
                "listener_id":listener_id,
                "success":success,
                "word_list":word_list,
                "obj_list":obj_list,
                "corr_obj":corr_obj,
                "choices_list":choices_list,
                "speaker_near_synonyms":[word['word'] for word in speaker_near_synonyms],
                "speaker_known_vocab":speaker_known_vocab
            }
        
    def run(self,rounds):
        save_flag=0
        for _ in range(rounds):
            save_flag+=1
            results=[]
            self.round+=1
            player_ids=[str(i) for i in range(self.player_num)]
            random.shuffle(player_ids)
            pairs=[]
            for i in range(0,self.player_num,2):
                pairs.append((player_ids[i],player_ids[i+1]))
            print(pairs)
            for pair in pairs:
                corr_obj=list(self.obj_loader.keys())[(self.round-1)%len(self.obj_loader)]
                keys_list=list(self.obj_loader.keys())
                keys_list.remove(corr_obj)
                random_keys=random.sample(keys_list,k=4)
                obj_dict={obj:self.obj_loader[obj] for obj in random_keys}
                result=self.communicate(
                    speaker_id=pair[0],
                    listener_id=pair[1],
                    obj_dict=obj_dict,
                )
                results.append(result)
            if save_flag%self.save_interval==0:
                self.save(results,True)
            else:
                self.save(results,False)
                
    def test_run(self,rounds):
        for _ in range(rounds):
            results=[]
            self.round+=1
            player_ids=[str(i) for i in range(self.player_num)]
            random.shuffle(player_ids)
            pairs=[]
            for i in range(0,self.player_num,2):
                pairs.append((player_ids[i],player_ids[i+1]))
            print(pairs)
            for pair in pairs:
                corr_obj=list(self.obj_loader.keys())[(self.round-1)%len(self.obj_loader)]
                keys_list=list(self.obj_loader.keys())
                keys_list.remove(corr_obj)
                random_keys=random.sample(keys_list,k=4)
                obj_dict={obj:self.obj_loader[obj] for obj in random_keys}
                result=self.test_communicate(
                    speaker_id=pair[0],
                    listener_id=pair[1],
                    obj_dict=obj_dict,
                )
                results.append(result)
            self.save(results,False)
                        
    def save(self,results,with_worddatabase):
        round_path=f"./sim_storage/{self.name}/round_{self.round}"
        try:
            os.makedirs(round_path,exist_ok=False)
        except:
            raise FileExistsError("The file for this round of the game already exists.")
        os.makedirs(os.path.join(round_path,"Communicator_Worddatabase"))
        with open(os.path.join(round_path,"results.json"),"w") as f:
            json.dump(results,f,indent=4)
        if with_worddatabase:
            for num in list(self.players.keys()):
                os.makedirs(os.path.join(round_path,"Communicator_Worddatabase",f"Player_{str(num)}"))
                self.players[num].save(
                    os.path.join(round_path,"Communicator_Worddatabase",f"Player_{str(num)}")
                )
        print(f"round_{self.round} has saved.")
    
    def load(self,round):
        if os.path.exists(f"./sim_storage/{self.name}/round_{round}"):
            pass
        else:
            raise FileExistsError('The folder for this round of the game does not exist.')
        self.round=round
        player_num=len(os.listdir(f"./sim_storage/{self.name}/round_{round}/Communicator_Worddatabase"))
        if player_num!=self.player_num:
            raise ValueError("The number of players set in the game is different from the historical number")
        for key in list(self.players.keys()):
            player_path=f"./sim_storage/{self.name}/round_{round}/Communicator_Worddatabase/Player_{str(key)}"
            self.players[key].load(player_path)
        print(f"round_{self.round} has loaded.")
        
        

                                      
class Referential_Game_wo_commu(Referential_Game):
    def __init__(self, name, player_num, letter_list, comm_num, save_interval, obj_loader, model_list):
        super().__init__(name, player_num, letter_list, comm_num, save_interval, obj_loader, model_list)
    def output(
        self,
        speaker_id:str,
        listener_id:str,
    ):
        speaker=self.players[speaker_id]
        listener=self.players[listener_id]
        word_list=[]
        corr_obj=list(self.obj_loader.keys())[(self.round-1)%len(self.obj_loader.keys())]
        corr_obj_features=self.obj_loader[corr_obj]
        corr_obj_embedding=text_embedding_request(corr_obj)
        speaker_known_vocab_num=speaker.word_database.search_word(**corr_obj_features)
        random.shuffle(speaker_known_vocab_num)
        speaker_known_vocab=[speaker.word_database.word_dict[num].word for num in speaker_known_vocab_num]
        speaker_near_synonyms=[speaker.word_database.word_dict[num].todict_wo_object() for num in speaker.word_database.search_near_synonyms(corr_obj)]
        speaker_short_memory=[]
       
        
        
        word=speaker.generate_new_word(
            vocab=speaker_near_synonyms,
            obj_features=corr_obj_features,
            failed_records=speaker_short_memory,
        )
        speaker.word_database.add_word(
                                text_embedding=corr_obj_embedding,
                                word=word,
                                obj=corr_obj,
                                **corr_obj_features,
                            )
        listener.word_database.add_word(
                                text_embedding=corr_obj_embedding,
                                word=word,
                                obj=corr_obj,
                                **corr_obj_features,
                            ) 
        word_list.append(word)
        return {
            "speaker_id":speaker_id,
            "listener_id":listener_id,
            "word_list":word_list,
            "corr_obj":corr_obj,
            "speaker_near_synonyms":[word['word'] for word in speaker_near_synonyms],
            "speaker_known_vocab":speaker_known_vocab,
        }
    def run(self, rounds):
        save_flag=0
        for _ in range(rounds):
            save_flag+=1
            results=[]
            self.round+=1
            player_ids=[str(i) for i in range(self.player_num)]
            random.shuffle(player_ids)
            pairs=[]
            for i in range(0,self.player_num,2):
                pairs.append((player_ids[i],player_ids[i+1]))
            print(pairs)
            for pair in pairs:
                result=self.output(
                    speaker_id=pair[0],
                    listener_id=pair[1],
                )
                results.append(result)
            if save_flag%self.save_interval==0:
                self.save(results,True)
            else:
                self.save(results,False)
        
        
        
    
    

    
    
    
    
       