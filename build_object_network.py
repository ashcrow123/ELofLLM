from data_loader.BRM_loader import load_object_feature_pairs_v2
from llm_methods.run_gpt_prompt import run_gpt_prompt_select_synonyms
import json
from tqdm import tqdm
from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-4.1-mini")
args = parser.parse_args()

model = args.model
all_pairs=load_object_feature_pairs_v2()
copy_loader=deepcopy(all_pairs)
BR_labels=[
        "smell",
        "sound",
        "tactile",
        "taste",
        "visual_colour",
        "visual_form_and_surface",
        "visual_motion",
        "function",
        "encyclopaedic",
        "taxonomic",
    ]
for concept,features in copy_loader.items():
    for label in features:
        if not (label in BR_labels):
            del all_pairs[concept][label]
copy_loader=deepcopy(all_pairs)
for concept,features in copy_loader.items():
    if all([features[label]==[] for label in BR_labels]):
        del all_pairs[concept]
all_words=list(all_pairs.keys())
results={key:[] for key in all_words}
count=0
for word in tqdm(all_words):
    object_features=all_pairs[word]
    del all_pairs[word]
    if not all_pairs:
        break
    features_list=list(all_pairs.values())
    words_list=list(all_pairs.keys())
    output=run_gpt_prompt_select_synonyms(object_features=object_features,
                                          features_list=features_list,
                                          model=model,
                                          verbose=False)
    
    num_list=output["num_list"]
    for num in num_list:
        results[word].append(words_list[num])
        results[words_list[num]].append(word)

with open(f"./data/{model.replace('/', '-')}_network.json","w") as f:
    json.dump(results,f,indent=4)




