# write_json.py
import json


json_w ={
    'num_labels':512,
    'use_score_num':10, # 0:False ( minimal number)
    'use_best':False,
    'batch_size':64,
    'epochs':40,
    'train_val':0.8,
    'metadata_path':'E:/work/danbooru/0/danbooru2021/metadata',
    'image_path':'C:/D/512px'
}

with open("project.json",'w',encoding='utf-8') as f:
    f.write(json.dumps(json_w)) 