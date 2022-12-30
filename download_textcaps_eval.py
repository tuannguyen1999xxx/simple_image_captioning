import json
import os
import random
from tqdm import tqdm
import requests
import wget
import urllib.request
import urllib.error

with open('nocaps_val_4500_captions.json','r',encoding='utf-8') as r:
    tmp = json.load(r)
    data = tmp['images']
    random.shuffle(data)
    img_ids = [x['id'] for x in data]
    # reference_strs = [x['reference_strs'] for x in data]
    flickr_original_url = [x['coco_url'] for x in data]
    anno = tmp['annotations']
dir = 'data/eval_nocaps'
os.makedirs(dir,exist_ok=True)

coco_anno = {"images":[],"annotations":[]}
count_cap = 0
count = 0
for c, image_url in tqdm(enumerate(flickr_original_url)):
    if count == 500:
        break
    img_id = img_ids[c]
    image_name = f'{str(img_id)}.jpg'
    reference_str = [x['caption'] for x in anno if x['image_id'] == img_id]

    img_frm = {'file_name':f'{dir[5:]}/{str(img_id)}.jpg','id':img_id}
    coco_anno["images"].append(img_frm)
    for re_str in reference_str:
        cap_frm = {'image_id':img_id,'caption':re_str,'id':count_cap}
        coco_anno["annotations"].append(cap_frm)
        count_cap += 1
    print(image_name)
    try:
        wget.download(image_url,os.path.join(dir,image_name))
        count += 1
    except urllib.error.HTTPError as err:
        print(err.code)
    # img_data = requests.get(image_url).content
    # with open(os.path.join(dir,image_name),'wb') as handler:
    #     handler.write(img_data)

with open('data/annotations/textcaps_anno_val_gt.json','w',encoding='utf-8') as w:
    json.dump(coco_anno,w,ensure_ascii=False,indent=4)

# print(img_ids[:10])
# print("------------------")
# print(reference_strs[:10])
# img_data = requests.get(image_url).content
# with open('image_name.jpg', 'wb') as handler:
#     handler.write(img_data)