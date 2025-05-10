from communicator.communicator import communicator
from llm_methods.gpt_structure import text_embedding_request
from data_loader.BRM_loader import BRM_loader
from data_loader.coco_loader import coco_loader
import random
import os
import json
def select_letters(num=16,seed=114514):
    random.seed(seed)
    all_letters=[]
    for i in "bcdfghjklmnpqrstvwxyz":
        for j in "aeiou":
            all_letters.append(i+j)
    letters=random.choices(all_letters,k=num)
    random.seed(None)
    return letters

#TODO 设置数据集
class game:
    def __init__(
        self,
        name:str,
        player_num:int,
        letter_list:list,
        comm_num,
        save_interval
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
            self.players[str(i)]=communicator(self.letter_list,num=str(i))
        self.round=0
        self.obj_loader=coco_loader()   
        self.obj_loader.sample(0.1)
        self.save_interval=save_interval
    def communicate(
                self,
                speaker_num:str,
                listener_num:str,
                obj_list:list):
        speaker=self.players[speaker_num]
        listener=self.players[listener_num]
        round=0
        success=False
        word_list=[]
        choices_list=[]
        corr_obj=random.choice(obj_list)
        corr_obj_embedding=text_embedding_request(corr_obj)
        obj_features_dict={obj:self.features_loader.generate_features(obj) for obj in obj_list}
        select_features_dict=dict()
        choices_dict=dict()
        choices="ABCDEFG"
        for i in range(len(obj_list)):
            choices_dict[choices[i]]=obj_list[i]
            select_features_dict[choices[i]]=obj_features_dict[obj_list[i]]
        speaker_known_vocab_num=speaker.word_database.search_word(**obj_features_dict[corr_obj])
        speaker_near_synonyms=[word.todict_wo_object() for word in speaker.word_database.search_near_synonyms(corr_obj_embedding)]
        speaker_short_memory=[]
        while True:
            round+=1
            if round<=self.comm_num:
                if speaker_known_vocab_num:
                    word_num=random.choice(speaker_known_vocab_num)
                    speaker_known_vocab_num.remove(word_num)
                    word=speaker.word_database.word_dict[word_num].word
                    choice=listener.listener_select(
                    word=word,
                    sf_dict=select_features_dict,
                    )
                    choices_list.append(choice)
                    if choices_dict[choice]==corr_obj:
                        success=True
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
                else:
                    
                    word=speaker.generate_new_word(
                        vocab=speaker_near_synonyms,
                        obj_features=obj_features_dict[corr_obj],
                        failed_records=speaker_short_memory,
                    )
                    choice=listener.listener_select(
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
            else:
                # speaker.word_database.add_word(
                #             text_embedding=corr_obj_embedding,
                #             word=word,
                #             obj=corr_obj,
                #             **obj_features_dict[corr_obj],
                #         )
                # listener.word_database.add_word(
                #     text_embedding=corr_obj_embedding,
                #     word=word,
                #     obj=corr_obj,
                #     **obj_features_dict[corr_obj],
                # )
                break
        return {
            "speaker_num":speaker_num,
            "listener_num":listener_num,
            "success":success,
            "word_list":word_list,
            "obj_list":obj_list,
            "corr_obj":corr_obj,
            "choices_list":choices_list,
            "speaker_near_synonyms":speaker_near_synonyms,
            "Known_vocab":[speaker.word_database.word_dict[num].word for num in speaker_known_vocab_num]
        }
    def run(self,rounds):
        save_flag=0
        for _ in range(rounds):
            save_flag+=1
            results=[]
            self.round+=1
            player_num=[str(i) for i in range(self.player_num)]
            random.shuffle(player_num)
            pairs=[]
            for i in range(0,self.player_num,2):
                pairs.append((player_num[i],player_num[i+1]))
            print(pairs)
            for pair in pairs:
                obj_list=self.obj_loader.get_random_caption()
                result=self.communicate(
                    speaker_num=pair[0],
                    listener_num=pair[1],
                    obj_list=obj_list,
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
        
        
                                
if __name__=="__main__":
    letter_list=select_letters()
    print(letter_list)
    g1=game(
        name="4test",
        player_num=10,
        letter_list=letter_list,
        comm_num=5,
        save_interval=3
    )
    g1.load(108)
    g1.run(12)
    #TODO
    '''
    1.为近义词搜索设置阈值√
    2.单词的遗忘机制（暂时先不做）
    3.打印数据流√
    4.object问题（设置提示词机制或换数据集）
    5.安全输出地报错机制（简单，优先解决）√
    6.很重要：重新思考word_to_key是否合适。
    7.speaker反复输出一个单词的问题。
    '''
    #需解决：大模型不用给定单词生成的问题。（换4o似乎已解决，待观察）
    
    
    
       