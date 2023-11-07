import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import image_loader


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class AvgStyleLoss(nn.Module):
    def __init__(self, target_paths, model):
        super(AvgStyleLoss, self).__init__()
        gram_matrixes = []
        # for target_feature in target_features:
        #     g = gram_matrix(target_feature).detach()
        #     gram_matrixes.append(g)
        # # average out the gram matrix
        # self.target = sum(gram_matrixes) / len(gram_matrixes)
        for target_path in target_paths:
            image = image_loader(target_path)
            feature = model(image).detach()
            g = gram_matrix(feature)
            gram_matrixes.append(g)
        self.target = sum(gram_matrixes) / len(gram_matrixes)
        print ("style shape: ", self.target.shape)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def gram_matrix(input):
    a,b, c, d = input.size() # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

