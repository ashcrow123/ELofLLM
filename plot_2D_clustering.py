from sklearn.manifold import MDS
from statistic.statistic_methods import *
import json
import matplotlib.pyplot as plt
from communicator.WordDatabase import WordDatabase
import seaborn as sns
from collections import defaultdict
import pandas as pd
# excel_path="data/McRae-BRM-InPress/CONCS_FEATS_concstats_brm.xlsx"
# df=pd.read_excel(excel_path)
# tax_dict=df["Feature"].value_counts().to_dict()
# pair_list=list(zip(tax_dict.keys(),tax_dict.values()))
# pair_list.sort(reverse=False,key=lambda pair:pair[-1])
path="sim_storage/cv_L14_A16_10/round_1600/Communicator_Worddatabase/Player_0"
# wd=WordDatabase(model="gpt-4.1-mini")
# tax_labels=list(dict(pair_list).keys())
# tax_labels=tax_labels[-25:-5]
# for i in range(len(tax_labels)):
#     tax_labels[i]=tax_labels[i].replace("_"," ")
# labels=[]
# wd.load(path)
# word_list=[wd.word_dict[i].word for i in wd.word_dict]
# for num,word in wd.word_dict.items():
#     taxs=word.toFeatures()
#     new_taxs=[]
#     for key,value in taxs.items():
#         new_taxs+=value
#     taxs=new_taxs
#     label="other"
#     for tax in tax_labels:
#         if tax in taxs:
#             label=tax
#             break
#     labels.append(label)
# dist_matrix=edit_matrix(word_list,word_list)
dist_matrix=edit_matrix(path=path)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
embedding = mds.fit_transform(dist_matrix)
sns.scatterplot(x=embedding[:,0],
                y=embedding[:,1],
                # hue=labels
                )
# for i, txt in enumerate(texts):
#     plt.annotate(f'{i}', (embedding[i,0], embedding[i,1]))
plt.title("MDS test")
plt.show()