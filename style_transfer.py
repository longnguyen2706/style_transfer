import glob
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from net.module import Normalization, ContentLoss, StyleLoss
import argparse

############################################## INIT ##############################################################
cnn = models.vgg19(pretrained=True).features.eval()

CONTENT_LAYER_DEFAULT = ['conv_4']
# STYLE_LAYER_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7']
STYLE_LAYER_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225]).to(device)


def image_loader(image_name):
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img,
                               content_layers=CONTENT_LAYER_DEFAULT, style_layers=STYLE_LAYER_DEFAULT):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img,
                       input_img, num_steps=300, style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_img, content_img)

    # we want to optimize the input and not the model parameters
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.data.clamp_(0, 1)

        style_score, content_score = 0, 0
        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
    return input_img, style_score.item()*style_weight, content_score.item()*content_weight


def train(cnn, style_img_path, content_img_path, output_img_path, metric_path, num_steps=300, style_weight=1000000, content_weight=1):

    style_img = image_loader(style_img_path)
    content_img = image_loader(content_img_path)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    plt.ion()
    input_img = content_img.clone()
    output, style_loss, content_loss  = run_style_transfer(cnn, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD,
                                content_img, style_img, input_img, num_steps=300, style_weight=1000000,
                                content_weight=1)
    print ("Style Loss: ", style_loss, "Content Loss: ", content_loss)
    # plt.figure()
    # imshow(output, title='Output Image')
    #
    # # sphinx_gallery_thumbnail_number = 4
    # plt.ioff()
    # plt.show()

    # save image
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    output = output.cpu().clone()
    output = output.squeeze(0)
    output = unloader(output)
    output.save(output_img_path)

    # save metric
    with open(metric_path, "w") as f:
        data = {}
        data["style_loss"] = style_loss
        data["content_loss"] = content_loss
        data["style_weight"] = style_weight
        data["content_weight"] = content_weight
        f.write(json.dumps(data))


if __name__ == '__main__':
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_weight", type=float, default=1000000)
    parser.add_argument("--content_weight", type=float, default=1)
    parser.add_argument("--dataset", type=str, default="monet")

    args = parser.parse_args()

    IMG_OUT_DIR = "./data/images"
    METRIC_OUT_DIR = "./data/metrics"
    print(args)
    if args.dataset == "monet":
        DATASET_PATH = "./datasets/midterm_report/"
        style_folder, content_folder = os.path.join(DATASET_PATH, 'style_images'), os.path.join(DATASET_PATH, 'content_images')
        # load all file in folder
        style_img_paths = sorted(glob.glob(style_folder + "/*.jpg"))
        content_img_paths = sorted(glob.glob(content_folder + "/*.png"))
        img_out_dir, metric_out_dir = os.path.join(IMG_OUT_DIR, "monet"), os.path.join(METRIC_OUT_DIR, "monet")       

       
    elif args.dataset == "cityscape":
        # style_folder, content_folder = os.path.join('./datasets/midterm_report/', 'style_images'), os.path.join('./datasets/cityscapes/testA')
        style_folder, content_folder = os.path.join('./datasets/cityscapes/testB'), os.path.join('./datasets/cityscapes/testA')
        style_img_paths = sorted(glob.glob(style_folder + "/*.jpg"))
        content_img_paths = sorted(glob.glob(content_folder + "/*.jpg"))
        img_out_dir, metric_out_dir = os.path.join(IMG_OUT_DIR, "cityscape_mask_single"), os.path.join(METRIC_OUT_DIR, "cityscape_mask_single")     

    print ("Style Image: ", len(style_img_paths), "Content Image: ", len(content_img_paths))  

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(metric_out_dir, exist_ok=True)

    for idx, content_img_path in enumerate(content_img_paths):

        content_image_name = content_img_path.split("/")[-1].split(".")[0]
        style_img_path = style_img_paths[idx]
        # if content_image_name in OUTLIER_FILES:
        output_img_path = os.path.join(img_out_dir,  "avg"+"_"+content_image_name+".jpg")
        metric_path =  os.path.join(metric_out_dir, "avg" + "_"+ content_image_name+".json")

        print( "Content Image: ", content_image_name)

        train(cnn, style_img_path, content_img_path, output_img_path, metric_path, num_steps=1000, style_weight=10000, content_weight=1)
