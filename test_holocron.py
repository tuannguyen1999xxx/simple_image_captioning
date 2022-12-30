import holocron.models as models

d53 = models.darknet53(pretrained=True)

print(d53)

# import torch
# import torch.nn as nn
#
# import torchvision.models as models
# from torchvision.models.vision_transformer import vit_b_16
# # Model
# class Yolov3_Encoder(nn.Module):
#     def __init__(self, embed_size):
#         super(Yolov3_Encoder, self).__init__()
#         model = torch.hub.load('ultralytics/yolov3', 'yolov3')
#
#         self.features = model.model
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(262395, embed_size)        # 262395: size of concate_vector from 3 outputs
#
#     def forward(self, x):
#         out = self.features(x)                           # out[1] have 3 outputs:[bs, 3, 28, 28, 85],[bs, 3, 14, 14, 85],[bs, 3, 7, 7, 85]
#         flatten_out = [self.flatten(z) for z in out]     # flatten all 3 outputs
#         concate_out = torch.cat(flatten_out,dim=1)        # concate 3 vectors [1, dimension]
#         vector_out = self.linear(concate_out)
#
#         return vector_out
#
# if __name__ == '__main__':
#     # x = torch.rand((3,3,224,224)).to(torch.device('cuda'))
#     # model = Yolov3_Encoder(512)
#     # model.to(torch.device('cuda'))
#     # model(x)
#
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s-seg')  # yolov5n - yolov5x6 or custom
#     print(model)
#     im = 'https://ultralytics.com/images/zidane.jpg'  # file, Path, PIL.Image, OpenCV, nparray, list
#     results = model(im)  # inference
#     results.print()