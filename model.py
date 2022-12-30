import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from model_darknet import darknet53
import numpy as np
import holocron.models as models_holo

class Yolov3_Encoder(nn.Module):
    def __init__(self, embed_size, num_heads=8, att=False):
        super(Yolov3_Encoder, self).__init__()
        model = torch.hub.load('ultralytics/yolov3', 'yolov3')
        # for p in model.parameters():
        #     p.requires_grad_(False)
        self.features = model.model
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(12495, embed_size), # 262395: size of concate_vector from 3 outputs
            nn.BatchNorm1d(embed_size),
            nn.ReLU()
        )
        self.linear_out = nn.Linear(embed_size, embed_size)
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.att = att
        print('Attention: ',att)
    def forward(self, x):
        # self.features.eval()
        with torch.no_grad():
            out = self.features(x)  # out[1] have 3 outputs:[bs, 3, 28, 28, 85],[bs, 3, 14, 14, 85],[bs, 3, 7, 7, 85]
        # out = self.features(x)
        # print(type(out), len(out))
        # print('0',type(out[0]), len(out[0]), out[0].shape)
        # print('-----------------------')
        # print('1',type(out[1]),len(out[1]),out[1][0].shape, out[1][1].shape, out[1][2].shape)
        # print('-----------------------')
        if self.training:
            out = [self.flatten(z) for z in out] # [bs, 199920], [bs, 49980], [bs, 12495]
        else:
            out = [self.flatten(z) for z in out[1]]

        # out = torch.cat(out,dim=1)
        out = self.linear(out[2])

        if self.att:

            out = self.attention(out, out, out)

            return out[0]
        else:

            return out

class Darknet_Encoder(nn.Module):
    def __init__(self,embed_size):
        super(Darknet_Encoder, self).__init__()
        self.model = darknet53(1000)
        checkpoint_path = '/home/usr/Workspaces/tuanns-ai/NLP/cnn_lstm/PyTorch-Darknet53/model_best.pth.tar'
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1000, embed_size)
    def forward(self,x):
        out = self.model(x)
        flatten_out = self.flatten(out)
        vector_out = self.linear(flatten_out)
        return vector_out
        
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, cnn='vgg'):
        """Create encoder by VGG16, load pretrained imagenet"""
        super(EncoderCNN, self).__init__()
        if cnn == 'vgg':
            print('Use backbone VGG16...')
            model = models.vgg16(pretrained=True)
            pooling = model.avgpool
            linear = nn.Linear(model.classifier[0].in_features, embed_size)

            self.features = list(model.features)
            self.features = nn.Sequential(*self.features)

            self.pooling = pooling
            self.flatten = nn.Flatten()
            # self.fc = model.classifier[0]
            self.linear = linear
        elif cnn == 'darknet':
            print('Use backbone Darknet53...')
            model = models_holo.darknet53(pretrained=True)
            linear = nn.Linear(model.classifier.in_features, embed_size)
            self.features = model.features
            self.pooling = model.pool
            self.flatten = nn.Flatten()
            self.linear = linear

    def forward(self, x):
        # with torch.no_grad():
        #     out = self.features(x)
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class CNN_LSTM_model(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20, att = False,state='train', cnn='vgg'):
        """Build model Encoder-decoder for image captioning task"""
        super(CNN_LSTM_model, self).__init__()
        if cnn == 'yolov3':
            self.encoder = Yolov3_Encoder(embed_size=embed_size, att=att)
        elif cnn == 'darknet53':
            self.encoder = Darknet_Encoder(embed_size=embed_size)
        else:
            self.encoder = EncoderCNN(embed_size=embed_size,cnn=cnn)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)
        self.state = state

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)

        return outputs

    def infer(self, images):
        features = self.encoder(images)
        outputs = self.decoder.sample(features)

        return outputs

if __name__ == "__main__":
    bs = 2
    nh = 8
    model = nn.MultiheadAttention(512, num_heads=nh)
    # q = torch.rand(bs, 3, 28, 28, 85).view(1)
    # k = torch.rand(bs, 3, 14, 14, 85).view(1)
    # v = torch.rand(bs, 3, 7, 7, 85).view(1)
    # q = torch.rand(2, 199920)
    # k = torch.rand(2,  49980)
    # v = torch.rand(2, 12495)
    #
    # sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
    # q = q.view(sz_b, len_q, 64)
    # k = k.view(sz_b, len_q,  64)
    # v = v.view(sz_b, len_q, 64)
    # q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    q = torch.rand(2, 512)
    # q = q.unsqueeze(1)
    # q = q.squeeze(1)
    out = model(q, q, q)
    print(out[0].shape)
    # print(q.shape)
    # k = torch.rand(2, 512)
    # v = torch.rand(2, 512)
    # ct, att = model(q,k,v)
    # print(ct)
    # print(ct.shape)
    # print('----------')
    # print(att.shape)
    # print(att)
    # x = torch.rand(2,3,224,224).to(torch.device('cuda'))
    # model = Yolov3_Encoder(512).to(torch.device('cuda'))
    # model.train()
    # model(x)
    # print(model)
    # print(models_holo.darknet53(pretrained=True))
    # model = EncoderDarknet(5)
    # model(x)
    # del model.Detect
    # print(model)
    # print(model)
    # img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    #
    # # Inference
    # results = model(img)
    #
    # # Results
    # results.print()