# Setting up the Environments
This project is a language emergence simulation based on LLM. To ensure the normal operation of the program, you need to complete some necessary configurations first. 
## 1. Configure OpenAI Api Key and Base Url
Please make sure that there are two system environment variables ""OPENAI_API_KEY"" and "OPENAI_API_BASE" in your system, which record your OpenAI api key and the base URL you use respectively. 
You can also modify the code in ```llm_methods/gpt_structure.py```:
```python
openai_api_key=os.environ.get("OPENAI_API_KEY") # Replace it with your api key
api_base = os.environ.get("OPENAI_API_BASE") # Replace it with your base URL
```
## 2. Install requirements.txt
Please create a Python environment first (Python version 3.12 is recommended).
Then, run the following code on your command line:
```
pip install -r requirements.txt
``` 
# Running Simulation
## Establishing semantic connections
In the simulation, the speaker agent generates words by obtaining semantically similar objects through semantic-based search. Before the referential game begins, you need to have LLM build and store the semantic relationship network of the object. You can run ```build_object_network.py``` to achieve it.
```
python build_object_network.py
```
You can modify the "model" parameter in this code to specify the LLM model. It is important to note that if you want to specify a model for the simulation agent in the referential game, please make sure that the semantic relationship network of these models has been built using ```build_object_network.py```.
## Running a Referential Game
Running the program is very simple. You just need to create a python file(you can name it "main.py"), copy the template below, and replace the settings with what you need.
```python 
from game import *
from data_loader.BRM_loader import load_object_feature_pairs
from copy import deepcopy
seed=40# random seed
letter_list=select_letters(num=<num>,seed=seed)# Replace <num> with the number of letters you want
obj_loader=load_object_feature_pairs(seed=seed,count=<count>) # Replace <count> with the number of objects you want for language emergence

model_list=[
    "gpt-4.1-mini",
    "gpt-4.1-mini",
]# You can also change to other models, but please ensure that the list size is consistent with the number of agents.

g1=Referential_Game(
name=<name>,# set the experiment name
    player_num=<num>,#set the agent num
    letter_list=letter_list,
    comm_num=3,# 3 attempts in one communication
    save_interval=4, #Save every four rounds
    obj_loader=obj_loader,
    max_length=<len>,# Set maximum word length
    model_list=model_list
)
g1.run(<round>)#Number of running rounds

train_loader=load_object_feature_pairs(seed=seed,count=400)
all_loader=load_object_feature_pairs()
obj_loader=dict()
for key,value in all_loader.items():
    if not (key in train_loader):
        obj_loader[key]=value
# Generate a generalization test set
g1=Referential_Game(
name=<name>,# set the experiment name
    player_num=<num>,#set the agent num
    letter_list=letter_list,
    comm_num=3,# 3 attempts in one communication
    save_interval=4, #Save every four rounds
    obj_loader=obj_loader,
    max_length=<len>,# Set maximum word length
    model_list=model_list
)
g1.load(<round>)# Load a round of experiments
rounds=len(obj_loader)
g1.test_run(rounds)# A generalization test, during which the agent's memory is not updated.
```

