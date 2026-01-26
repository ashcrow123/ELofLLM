import os
os.system("set NO_PROXY=*")
from game import *
from data_loader.BRM_loader import get_splited_data,get_splited_data_wo_feature
import sys
import random
exp_num=1
seed=39 + exp_num
letter_num=48
max_length=6
name=f"cv_L{max_length}_A{letter_num}_{exp_num}_gen2"
sys.stdout = open(f'output/{name}.txt', 'a', encoding='utf-8')
letter_list=select_letters(num=letter_num,seed=seed)
# attr_feature=[
#     ("function","is edible"),
#     ("taxonomic","an animal"),
#     ("visual_colour","different colours"),
#     ("visual_colour","is brown"),
#     ("visual_form_and_surface","made of metal"),
# ]
# (attr,feature)=attr_feature[seed%len(attr_feature)]
train_loader,test_loader=get_splited_data(count=400,seed=seed)
# train_loader_copy=dict()
# random.seed(seed)
# keys=random.sample(list(train_loader.keys()),k=300)
# for k in keys:
#     train_loader_copy[k]=train_loader[k]
# train_loader=train_loader_copy
model_list=[
    "gpt-4.1-mini",
    "gpt-4.1-mini",
]

g1=Referential_Game(
name=name,
    player_num=2,
    letter_list=letter_list,
    comm_num=3,
    save_interval=4,
    obj_loader=train_loader,
    max_length=max_length,
    model_list=model_list
)
g1.run(8)

# g1=Referential_Game(
#     name=name,
#     player_num=2,
#     letter_list=letter_list,
#     comm_num=3,
#     save_interval=4,
#     obj_loader=test_loader,
#     model_list=model_list,
#     max_length=max_length
# )
# g1.load(4*len(train_loader))
# rounds=len(test_loader)
# g1.test_run(rounds)
