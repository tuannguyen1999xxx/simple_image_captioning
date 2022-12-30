import argparse
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
import json
from evaluation import *
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # dir_save = 'out_caption_results_dn53_new'
    dir_save = args.dir_save
    os.makedirs(dir_save, exist_ok=True)

    best = 0
    best_epoch = 0

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        # transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    data_val_loader = get_loader(args.eval_dir, args.eval_path, vocab,
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,state='val')
    # Build the models
    print(len(vocab))
    model = CNN_LSTM_model(args.embed_size, args.hidden_size, len(vocab), args.num_layers,att=args.att, cnn=args.cnn).to(device)
    if args.checkpoint_path is not None:
        print("Load checkpoint...")
        model.load_state_dict(torch.load(args.checkpoint_path))
        model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Params: ",pytorch_total_params)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        model.train()
        for i, (images, captions, lengths) in tqdm(enumerate(data_loader)):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            outputs = model(images, captions, lengths)
            loss = criterion(outputs, targets)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))
            # if i == 1:
            #     break
        # Save the model checkpoints
        save_name = f'caption_epoch{epoch}.json'
        coco_val = eval_step(model, data_val_loader, vocab,
                                path_anno_eval=args.eval_path, path_save=os.path.join(dir_save, save_name),batch_size=args.batch_size)

        if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
            best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
            best_epoch = epoch
            torch.save(model.state_dict(),os.path.join(args.model_path,f'checkpoint_best.pth'))

        log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                         'epoch': epoch,
                         'best_epoch': best_epoch}
        with open(os.path.join(dir_save,'log_train.txt'),'a') as w:
            w.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/train2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--eval_path', type=str, default='data/annotations/captions_val2014.json')
    parser.add_argument('--eval_dir', type=str, default='data/val2014')

    parser.add_argument('--dir_save', type=str, default='out_caption_results')
    parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    parser.add_argument('--att', type=bool, default=False)
    parser.add_argument('--cnn',choices=['vgg','darknet', 'yolov3'], type=str, default='vgg', help='backbone use for extracting feature')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()
    print(args)
    main(args)