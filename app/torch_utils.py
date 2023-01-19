# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import io
# from PIL import Image


# #load our model
# #function to tranform image to tensor
# #function to predict

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.input_size = input_size
#         self.l1 = nn.Linear(input_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, num_classes)  
    
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         # no activation and no softmax at the end
#         return out

# input_size = 784 # 28x28
# hidden_size = 500 
# num_classes = 10
# model = NeuralNet(input_size, hidden_size, num_classes)

# PATH = "mnist_ffn.pth"
# model.load_state_dict(torch.load(PATH))
# model.eval()

# # image -> tensor
# def transform_image(image_bytes):
#     transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                     transforms.Resize((28,28)),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.1307,),(0.3081,))])

#     image = Image.open(io.BytesIO(image_bytes))
#     return transform(image).unsqueeze(0)

# # predict
# def get_prediction(image_tensor):
#     images = image_tensor.reshape(-1, 28*28)
#     outputs = model(images)
#         # max returns (value ,index)
#     _, predicted = torch.max(outputs.data, 1)
#     return predicted

import torch, os, io
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']

def get_vgg_layers(config, batch_norm):

    layers = []
    in_channels = 3
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)

vgg16_layers = get_vgg_layers(vgg16_config, batch_norm=True)

OUTPUT_DIM = 5
model = VGG(vgg16_layers, OUTPUT_DIM)

PATH = "model6.pt"
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()


classes = ['Varroa, Small Hive Beetles', 'ant problems', 'few varrao, hive beetles', 'healthy', 'hive being robbed', 'missing queen']

def transform_image(image_bytes):

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    transformer = transforms.Compose([
                      transforms.Resize([pretrained_size,pretrained_size]),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=pretrained_means,
                      std=pretrained_stds)
    ])
    image = Image.open(io.BytesIO(image_bytes))
    # image.save()
    image_tensor=transformer(image).float()
    image_tensor=image_tensor.unsqueeze_(0)
    image_tensor=image_tensor.to(device)
    return image_tensor

def get_prediction(img_tensor):
    
    classes = ['Varroa, Small Hive Beetles', 'ant problems', 'few varrao, hive beetles', 'healthy', 'hive being robbed', 'missing queen']

    if torch.cuda.is_available():
        img_tensor.cuda()
    input=Variable(img_tensor)
    y_pred,_=model(input)
    index=y_pred.cpu().data.numpy().argmax()
    pred=classes[index]

    return pred
    
    
  
  
  