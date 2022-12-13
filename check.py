import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url
from tqdm import tqdm

def coco_caption_eval(annotation_file, results_file):

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval
def save_result(out_format, path_save):

    new_out = []
    print("Save results...")
    for x in tqdm(out_format):
        ids = [x["image_id"] for x in new_out]
        if x["image_id"] not in ids:
            new_out.append(x)
    with open(path_save,'w',encoding='utf-8') as w:
        json.dump(new_out,w,ensure_ascii=False,indent=4)
coco_caption_eval('/home/usr/Workspaces/tuanns-ai/NLP/cnn_lstm/data/annotations/captions_val2014.json','/home/usr/Workspaces/tuanns-ai/NLP/cnn_lstm/tmp.json')
# with open('/home/usr/Workspaces/tuanns-ai/NLP/cnn_lstm/out_caption_results/caption_epoch0.json','r',encoding='utf-8') as r:
#     data = json.load(r)
#
# save_result(data, 'tmp.json')