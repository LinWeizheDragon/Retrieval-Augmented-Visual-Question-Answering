import pandas as pd
import ast
import json
import base64
import os
import numpy as np
np.set_printoptions(suppress=True, precision=4)


vinvl_dataset_tsv = '../materials/scene_graph_benchmark/datasets/fvqa/test.hw.tsv'
prefix = 'test'
label_tsv_name = f"{prefix}.label.tsv"
feature_tsv_name = f"{prefix}.feature.tsv"
yaml_file_name = f"{prefix}.yaml"

hw_df = pd.read_csv(vinvl_dataset_tsv,sep='\t',header=None,converters={1:ast.literal_eval},index_col=0)

print(hw_df.head())

sg_tsv = '../data/fvqa/pre-extracted_features/vinvl_output/vinvl_fvqa_testset/inference/vinvl_vg_x152c4/predictions.tsv'
df = pd.read_csv(sg_tsv,sep='\t',header = None,converters={1:json.loads})#converters={1:ast.literal_eval})
df[1] = df[1].apply(lambda x: x['objects'])

#example
print(df.loc[0,1][0])

def generate_additional_features(rect,h,w):
    mask = np.array([w,h,w,h],dtype=np.float32)
    rect = np.clip(rect/mask,0,1)
    res = np.hstack((rect,[rect[3]-rect[1], rect[2]-rect[0]]))
    return res.astype(np.float32)

def generate_features(x):
    idx, data,num_boxes = x[0],x[1],len(x[1])
    h,w,features_arr = hw_df.loc[idx,1][0]['height'],hw_df.loc[idx,1][0]['width'],[]

    for i in range(num_boxes):
        features = np.frombuffer(base64.b64decode(data[i]['feature']),np.float32)
        pos_feat = generate_additional_features(data[i]['rect'],h,w)
        x = np.hstack((features,pos_feat))
        features_arr.append(x.astype(np.float32))
        
    features = np.vstack(tuple(features_arr))
    features = base64.b64encode(features).decode("utf-8")
    return {"features":features, "num_boxes":num_boxes}

def generate_labels(x):
    data = x[1]
    res = [{"class":el['class'].capitalize(),"conf":el['conf'], "rect": el['rect']} for el in data] 
    return res


df['feature'] = df.apply(generate_features,axis=1)
df['feature'] = df['feature'].apply(json.dumps)

df['label'] = df.apply(generate_labels,axis=1)
df['label'] = df['label'].apply(json.dumps)

print(df.head())


OUTPUT_DIR = '../materials/scene_graph_benchmark/datasets/fvqa_for_oscar'
LABEL_FILE = os.path.join(OUTPUT_DIR, label_tsv_name)
FEATURE_FILE = os.path.join(OUTPUT_DIR, feature_tsv_name)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"path to {OUTPUT_DIR} created")

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
tsv_writer(df[[0,'label']].values.tolist(),LABEL_FILE)
tsv_writer(df[[0,'feature']].values.tolist(),FEATURE_FILE)

import yaml
import os.path as op

yaml_dict = {"label": label_tsv_name,
             "feature": feature_tsv_name}

with open(op.join(OUTPUT_DIR, yaml_file_name), 'w') as file:
    yaml.dump(yaml_dict, file)