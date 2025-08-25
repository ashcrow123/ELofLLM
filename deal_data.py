import os
import json
from collections import defaultdict
import pandas as pd
def update_word_dicts(root_path: str, reference_dict: dict):
    """
    在 root_path 目录下递归查找所有名为 word_dict.json 的文件，
    原地修改其中每个值字典的 "taxonomic" 字段，其值来自 reference_dict[obj]
    """
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == "word_dict.json":
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        word_dict = json.load(f)

                    modified = False

                    for key, value in word_dict.items():
                        if isinstance(value, dict) and "obj" in value:
                            obj_key = value["obj"]
                            if obj_key in reference_dict:
                                value["taxonomic"] = reference_dict[obj_key]
                                modified = True

                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(word_dict, f, ensure_ascii=False, indent=4)
                        print(f"✅ 修改成功：{file_path}")
                    else:
                        print(f"ℹ️ 无需修改：{file_path}")

                except Exception as e:
                    print(f"❌ 处理失败：{file_path}，错误：{e}")

# 使用示例
if __name__ == "__main__":
    # input_path = "your/path/to/folder"  # 替换为实际路径
    # input_dict = {
    #     "apple": "fruit",
    #     "cat": "animal",
    #     "python": "language"
    # }
    # update_word_dicts(input_path, input_dict)
    input_path="sim_storage/with_gpt_word_network_copy/round_1600"
    input_dict=defaultdict(list)
    path="data/McRae-BRM-InPress/CONCS_FEATS_concstats_brm.xlsx"
    df=pd.read_excel(path)
    df['Feature'] = df['Feature'].str.replace('_', ' ')
    df['BR_Label'] = df['BR_Label'].str.replace('-', '_')
    df['Feature'] = df['Feature'].str.replace(r'^beh - ', 'living behavior: ', regex=True)
    df['Feature'] = df['Feature'].str.replace(r'^inbeh - ', 'non-living behavior: ', regex=True)
    for index,row in df.iterrows():
        if row["BR_Label"]=="taxonomic":
            input_dict[row["Concept"]].append(row["Feature"])
    update_word_dicts(input_path,input_dict)
    
