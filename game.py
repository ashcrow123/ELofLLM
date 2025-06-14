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
        obj_loader
    ):
        self.name=name
        self.comm_num=comm_num
        if player_num<2 or player_num%2!=0:
            raise ValueError("The number of players must be an even number and greater than 1.")
        os.makedirs(f"./sim_storage/{self.name}",exist_ok=True)
        self.features_loader=BRM_loader()
        self.features_loader.load_embedding("BRM")
        self.player_num=player_num 
        self.letter_list=letter_list
        self.players=dict()
        for i in range(player_num):
            self.players[str(i)]=communicator(self.letter_list,id=str(i))
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
        # obj_features_dict={obj:self.features_loader.generate_features(obj) for obj in obj_list}
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
        speaker_near_synonyms=[speaker.word_database.word_dict[num].todict_wo_object() for num in speaker.word_database.search_near_synonyms(corr_obj,corr_obj_embedding)]
        speaker_short_memory=[]
        while True: 
            round+=1
            if speaker_known_vocab_num:
                weights=speaker.word_database.weight_output(speaker_known_vocab_num,"speaker")
                for i in reversed(range(len(weights))):
                    if weights[i]<=0.1:
                        speaker.word_database.delete(speaker_known_vocab_num[i])
                        speaker_known_vocab_num.pop(i)
                        speaker_known_vocab.pop(i)
            if round<=self.comm_num:
                if speaker_known_vocab_num:
                    # word_num=random.choice(speaker_known_vocab_num)
                    weights=speaker.word_database.weight_output(speaker_known_vocab_num,"speaker")
                    word_num=random.choices(speaker_known_vocab_num,weights=weights,k=1)[0]
                    speaker_known_vocab_num.remove(word_num)
                    word=speaker.word_database.word_dict[word_num].word
                    word_exists,choice,used_num=listener.listener_select(
                    word=word,
                    sf_dict=select_features_dict,
                    )
                    choices_list.append(choice)
                    if choices_dict[choice]==corr_obj:
                        success=True
                        speaker.word_database.word_dict[word_num].speak_fail_count-=1 if speaker.word_database.word_dict[word_num].speak_fail_count>-3 else 0
                        if not word_exists:
                            listener.word_database.add_word(
                                text_embedding=corr_obj_embedding,
                                word=word,
                                obj=corr_obj,
                                **obj_features_dict[corr_obj],
                            )
                        else:
                            listener.word_database.word_dict[used_num].listen_fail_count-=1 if listener.word_database.word_dict[used_num].listen_fail_count>-3 else 0
                        word_list.append(word)
                        break
                    else:
                        short_memory={
                            word:obj_features_dict[choices_dict[choice]],
                        }
                        speaker.word_database.word_dict[word_num].speak_fail_count+=1
                        if speaker.word_database.word_dict[word_num].speak_fail_count>=3:
                            speaker.word_database.delete(word_num)
                        speaker_short_memory.append(short_memory)
                        if word_exists:
                            listener.word_database.word_dict[used_num].listen_fail_count+=1
                            if listener.word_database.word_dict[used_num].listen_fail_count>=3:
                                listener.word_database.delete(used_num)
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
                        speaker.word_database.add_word(
                            text_embedding=corr_obj_embedding,
                            word=word,
                            obj=corr_obj,
                            **obj_features_dict[corr_obj],
                        )
                        if not word_exists:
                            listener.word_database.add_word(
                                text_embedding=corr_obj_embedding,
                                word=word,
                                obj=corr_obj,
                                **obj_features_dict[corr_obj],
                            )
                        else:
                            listener.word_database.word_dict[used_num].listen_fail_count-=1 if listener.word_database.word_dict[used_num].listen_fail_count>-3 else 0
                        word_list.append(word)
                        break
                    else:
                        short_memory={
                            word:obj_features_dict[choices_dict[choice]],
                        }
                        speaker_short_memory.append(short_memory)
                        if word_exists:
                            listener.word_database.word_dict[used_num].listen_fail_count+=1
                            if listener.word_database.word_dict[used_num].listen_fail_count>=3:
                                listener.word_database.delete(used_num)
                        word_list.append(word)
            else:
                # if not (word in speaker_known_vocab):
                #     speaker.word_database.add_word(
                #                 text_embedding=corr_obj_embedding,
                #                 word=word,
                #                 obj=corr_obj,
                #                 **obj_features_dict[corr_obj],
                #             )
                # if not word_exists:
                #     listener.word_database.add_word(
                #         text_embedding=corr_obj_embedding,
                #         word=word,
                #         obj=corr_obj,
                #         **obj_features_dict[corr_obj],
                #     )
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
        
        
class one_word_game(Referential_Game): 
    def __init__(self, 
                 name, 
                 player_num, 
                 letter_list, 
                 comm_num, 
                 save_interval,
                 obj_loader,
                 seed):
        super().__init__(name, player_num, letter_list, comm_num, save_interval,obj_loader)
        random.seed(seed)
        self.random_object=random.choice(list(self.obj_loader.keys()))
        self.object_pair={self.random_object:self.obj_loader[self.random_object]}
        random.seed(None)
    def communicate( 
                self,
                speaker_num:str,
                listener_num:str,
                obj_dict:dict):
        speaker=self.players[speaker_num]
        listener=self.players[listener_num]
        round=0
        success=False
        word_list=[]
        choices_list=[]
        # obj_list=list(obj_dict.keys())
        # corr_obj=random.choice(obj_list)
        corr_obj=self.random_object
        corr_obj_embedding=text_embedding_request(corr_obj)
        # obj_features_dict={obj:self.features_loader.generate_features(obj) for obj in obj_list}
        obj_dict={**obj_dict,**self.object_pair}
        items=list(obj_dict.items())
        random.shuffle(items)
        obj_dict=dict(items)
        obj_list=list(obj_dict.keys())
        obj_features_dict=obj_dict
        select_features_dict=dict()
        choices_dict=dict()
        choices="ABCDE"
        for i in range(len(obj_list)):
            choices_dict[choices[i]]=obj_list[i]
            select_features_dict[choices[i]]=obj_features_dict[obj_list[i]]
        speaker_known_vocab_num=speaker.word_database.search_word(**obj_features_dict[corr_obj])
        random.shuffle(speaker_known_vocab_num)
        speaker_known_vocab=[speaker.word_database.word_dict[num].word for num in speaker_known_vocab_num]
        speaker_near_synonyms=[word.todict_wo_object() for word in speaker.word_database.search_near_synonyms(corr_obj,corr_obj_embedding)]
        speaker_short_memory=[]
        while True: 
            round+=1
            if speaker_known_vocab_num:
                weights=speaker.word_database.weight_output(speaker_known_vocab_num,"speaker")
                for i in reversed(range(len(weights))):
                    if weights[i]<=0.1:
                        speaker.word_database.delete(speaker_known_vocab_num[i])
                        speaker_known_vocab_num.pop(i)
                        speaker_known_vocab.pop(i)
            if round<=self.comm_num:
                if speaker_known_vocab_num:
                    # word_num=random.choice(speaker_known_vocab_num)
                    weights=speaker.word_database.weight_output(speaker_known_vocab_num,"speaker")
                    word_num=random.choices(speaker_known_vocab_num,weights=weights,k=1)[0]
                    speaker_known_vocab_num.remove(word_num)
                    word=speaker.word_database.word_dict[word_num].word
                    word_exists,choice,used_num=listener.listener_select(
                    word=word,
                    sf_dict=select_features_dict,
                    )
                    choices_list.append(choice)
                    if choices_dict[choice]==corr_obj:
                        success=True
                        speaker.word_database.word_dict[word_num].speak_fail_count-=1 if speaker.word_database.word_dict[word_num].speak_fail_count>-3 else 0
                        if not word_exists:
                            listener.word_database.add_word(
                                text_embedding=corr_obj_embedding,
                                word=word,
                                obj=corr_obj,
                                **obj_features_dict[corr_obj],
                            )
                        else:
                            listener.word_database.word_dict[used_num].listen_fail_count-=1 if listener.word_database.word_dict[used_num].listen_fail_count>-3 else 0
                        word_list.append(word)
                        break
                    else:
                        short_memory={
                            word:obj_features_dict[choices_dict[choice]],
                        }
                        speaker.word_database.word_dict[word_num].speak_fail_count+=1
                        if speaker.word_database.word_dict[word_num].speak_fail_count>=3:
                            speaker.word_database.delete(word_num)
                        speaker_short_memory.append(short_memory)
                        if word_exists:
                            listener.word_database.word_dict[used_num].listen_fail_count+=1
                            if listener.word_database.word_dict[used_num].listen_fail_count>=3:
                                listener.word_database.delete(used_num)
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
                        speaker.word_database.add_word(
                            text_embedding=corr_obj_embedding,
                            word=word,
                            obj=corr_obj,
                            **obj_features_dict[corr_obj],
                        )
                        if not word_exists:
                            listener.word_database.add_word(
                                text_embedding=corr_obj_embedding,
                                word=word,
                                obj=corr_obj,
                                **obj_features_dict[corr_obj],
                            )
                        else:
                            listener.word_database.word_dict[used_num].listen_fail_count-=1 if listener.word_database.word_dict[used_num].listen_fail_count>-3 else 0
                        word_list.append(word)
                        break
                    else:
                        short_memory={
                            word:obj_features_dict[choices_dict[choice]],
                        }
                        speaker_short_memory.append(short_memory)
                        if word_exists:
                            listener.word_database.word_dict[used_num].listen_fail_count+=1
                            if listener.word_database.word_dict[used_num].listen_fail_count>=3:
                                listener.word_database.delete(used_num)
                        word_list.append(word)
            else:
                # if not (word in speaker_known_vocab):
                #     speaker.word_database.add_word(
                #                 text_embedding=corr_obj_embedding,
                #                 word=word,
                #                 obj=corr_obj,
                #                 **obj_features_dict[corr_obj],
                #             )
                # if not word_exists:
                #     listener.word_database.add_word(
                #         text_embedding=corr_obj_embedding,
                #         word=word,
                #         obj=corr_obj,
                #         **obj_features_dict[corr_obj],
                #     )
                break
        return {
            "speaker_num":speaker_num,
            "listener_num":listener_num,
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
                # obj_list=self.obj_loader.get_random_caption()
                random_keys=random.sample(list(self.obj_loader.keys()),k=4)
                obj_dict={obj:self.obj_loader[obj] for obj in random_keys}
                result=self.communicate(
                    speaker_num=pair[0],
                    listener_num=pair[1],
                    obj_dict=obj_dict,
                )
                results.append(result)
            if save_flag%self.save_interval==0:
                self.save(results,True)
            else:
                self.save(results,False)
                                      
if __name__=="__main__":
    letter_list=select_letters(num=10,seed=42)
    obj_loader=load_object_feature_pairs(42,400)
    print(letter_list)
    g1=Referential_Game(
        name="2p_l42_o42_w400",
        player_num=2,
        letter_list=letter_list,
        comm_num=3,
        save_interval=4,
        obj_loader=obj_loader
    )
    g1.load(400)
    g1.run(400)
    
    

    
    
    
    
       