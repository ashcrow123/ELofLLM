from data_loader.coco_loader import coco_loader
from llm_methods.gpt_structure import *
from concurrent.futures import ThreadPoolExecutor
import random
import json
import os
from llm_methods.run_gpt_prompt import run_gpt_prompt_wo_vocab, run_gpt_prompt_with_vocab

class VocabError(RuntimeError):
    pass

class Sender:
    def __init__(self,vocab_limit:int,token_limit:int=5,thread_num:int=10):
        self.token_limit=token_limit
        self.vocab_limit=vocab_limit
        self.thread_num=thread_num
        self.vocab=set()
    def sample_words(self,prompt):
        if self.vocab_limit>len(self.vocab):
            if self.thread_num==1:
                output= run_gpt_prompt_wo_vocab(prompt,self.token_limit)
                _,token_list=tokens_check(output)
                self.vocab.update(token_list)
                if len(self.vocab)>=self.vocab_limit:
                    self.vocab=set(list(self.vocab)[:self.vocab_limit])
                    vocab_dict=dict()
                    for key,value in enumerate(self.vocab):
                        vocab_dict[key]=value
                    self.vocab=vocab_dict
                result=[output]
            elif self.thread_num>1:
                args=[(p,self.token_limit) for p in prompt]
                with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
                    futures=[executor.submit(run_gpt_prompt_wo_vocab,*arg) for arg in args]
                    result=[]
                    for future in futures:
                        output=future.result()
                        result.append(output)
                        _,token_list=tokens_check(output)
                        self.vocab.update(token_list)
                    if len(self.vocab)>=self.vocab_limit:
                        self.vocab=list(self.vocab)[:self.vocab_limit]
                        vocab_dict=dict()
                        for key,value in enumerate(self.vocab):
                            vocab_dict[key]=value
                        self.vocab=vocab_dict
            else:
                raise ValueError("The number of threads must be positive.")
        else:
            raise VocabError("The vocabulary limit has been reached.")
        return result
    
    def generate_message(self,prompt):
        if self.vocab_limit==len(self.vocab):
            if self.thread_num==1:
                output= run_gpt_prompt_with_vocab(prompt,self.token_limit,self.vocab)
                result=[output]
            elif self.thread_num>1:
                args=[(p,self.token_limit,self.vocab) for p in prompt]
                with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
                    futures=[executor.submit(run_gpt_prompt_with_vocab,*arg) for arg in args]
                    result=[]
                    for future in futures:
                        output=future.result()
                        result.append(output)
            else:
                raise ValueError("The number of threads must be positive.")
        else:
            raise RuntimeError("The vocabulary limit has not been reached. Please sample words first.")
        return result
        
    def get_vocab(self): 
        return self.vocab
    
    def get_vocab_limit(self):
        return self.vocab_limit

class caption_game:
    def __init__(self,
                 name:int,
                 data_loader,
                 sample_rate,
                 sender:Sender,
                 batch_size:int=10):
        
        self.name=name
        self.batch_size=batch_size
        self.sender=sender
        self.data_loader=data_loader
        self.sample_rate=sample_rate
        self.caption_result=dict()
        
    def get_caption_result(self):
        return self.caption_result
    
    def play(self):
        self.data_loader.sample(self.sample_rate)
        image_ids=self.data_loader.get_image_ids()
        for i in range(len(image_ids)//self.batch_size+1):
            try:
                ids=image_ids[i*self.batch_size:(i+1)*self.batch_size]
            except:
                ids=image_ids[i*self.batch_size:]
            captions=[]
            captions_ids=[]
            for id in ids:
                caption=random.sample(self.data_loader.get_captions(id),1)
                captions.append(caption[0]["caption"])
                captions_ids.append(caption[0]["id"])
                
            try:
                results=self.sender.sample_words(captions)
            except Exception as VocabError:
                results=self.sender.generate_message(captions)
            
            
            print(f"---------------------------Batch_{i}-------------------------------")
            if any("Error:The output of GPT is illegal" in result for result in results):
                print("WARNING: The output of GPT is illegal. ")
            else:
                print("The output of GPT is normal.")            
            print("-----------------------------------------------------------------")
            new_data=list(zip(ids,captions_ids,captions,results))
            for (a,b,c,d) in new_data:
                # print(f"image_id:{a}, caption_id:{b}, caption:{c}, gpt_response:{d}\n")
                self.caption_result[a]={
                    "caption_id":b,
                    "caption":c,
                    "gpt_response":d
                }
        return
    
    def save(self):
        path=os.path.join("./storage",self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"{path} already exists.")
        with open(os.path.join(path,"caption.json"),"w") as f:
            json.dump(self.caption_result,f,indent=2)
        with open(os.path.join(path,"vocab.json"),"w") as f:
            json.dump(self.sender.get_vocab(),f,indent=2)
        print(f"Saved to {path}")
        return
        

if __name__=="__main__":
    start_time=time.time()
    data_loader=coco_loader(task="caption",dataset="train")
    sender=Sender(
        vocab_limit=256,
        token_limit=5,
        thread_num=12,
        )
    
    game=caption_game(
        name="test",
        data_loader=data_loader,
        sample_rate=0.3,
        sender=sender,
        batch_size=10
    )
    
    game.play()
    game.save()
    end_time=time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    
    


    