import argparse
import json

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, CNN_LSTM_model
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tqdm import tqdm
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url


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

def from_id_to_sentence(word_ids, vocab):
    sampled_caption = []
    for word_id in word_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            return sampled_caption

def batch_ids_to_batch_sentences(sampled_ids, vocab):
    sentences = []

    for sampled_id in sampled_ids:

        sampled_id_out = sampled_id.cpu().numpy()
        sampled_caption = from_id_to_sentence(sampled_id_out, vocab)
        if sampled_caption is not None:
            sentence = ' '.join(sampled_caption)
        else:
            sentence = ''
        sentences.append(sentence)

    return sentences

def save_result(out_format, path_save):

    new_out = []
    print("Save results...")
    for x in tqdm(out_format):
        ids = [x["image_id"] for x in new_out]
        if x["image_id"] not in ids:
            new_out.append(x)
    with open(path_save,'w',encoding='utf-8') as w:
        json.dump(new_out,w,ensure_ascii=False,indent=4)

def eval_step(model, data_loader, vocab, path_anno_eval, path_save, batch_size):

    out_format = []
    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        for i, (images, captions, lengths, img_ids) in tqdm(enumerate(data_loader)):
            # Set mini-batch dataset
            print(f"Gen caption batch {i}/{int(len(data_loader)/batch_size)}...")

            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            outputs = model(images, captions, lengths)
            loss = criterion(outputs, targets)
            sampled_ids = model.infer(images)

            sentences = [x.strip('<start>').strip('<end>') for x in batch_ids_to_batch_sentences(sampled_ids, vocab)]
            formats = [{"image_id":img_ids[i],"caption":sentences[i]} for i in range(len(img_ids))]

            out_format.extend(formats)
            total_loss += loss

    save_result(out_format, path_save)
    coco_eval = coco_caption_eval(path_anno_eval,path_save)

    return coco_eval

def main(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((args.crop_size,args.crop_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    dir_save = 'out'
    os.makedirs(dir_save,exist_ok=True)
    save_name = 'out.json'
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.eval_path, vocab,
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,state='val')

    model = CNN_LSTM_model(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    model.load_state_dict(torch.load('./models/checkpoint_037.pth'))

    eval_step(model, data_loader, vocab, path_anno_eval=args.eval_path,path_save=os.path.join(dir_save,save_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/flick_anno_val_gt.json',
                        help='path for train annotation json file')
    parser.add_argument('--eval_path', type=str, default='data/annotations/flick_anno_val_gt.json')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)