import argparse
import glob
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from net.module import Normalization, ContentLoss, AvgStyleLoss
from utils.utils import get_device, image_loader
import config.avg_modified_config as mod_config
import config.avg_original_config as ori_config

############################################## INIT ##############################################################
cnn = models.vgg19(pretrained=True).features.eval()

CONTENT_LAYER_DEFAULT = ['conv_4']
STYLE_LAYER_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

IMG_OUT_DIR = "./data/images"
METRIC_OUT_DIR = "./data/metrics"


device = get_device()
# torch.cuda.set_device(device)

CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225]).to(device)

# OUTLIER_FILES = ['2014-08-20 08:13:24_real', '2013-09-09 15:09:50_real', '2014-08-24 18:28:05_real',
#                  '2014-05-24 03:00:10_real', '2014-07-01 23:36:11_real', '2014-08-25 21:07:24_real',
#                  '2014-07-19 04:26:01_real', '2014-08-27 00:26:02_real', '2014-08-28 14:24:51_real',
#                  '2014-08-05 16:20:33_real', '2012-05-15 07:40:41_real', '2014-08-05 17:32:46_real',
#                  '2014-08-31 07:26:58_real', '2014-08-31 09:08:36_real', '2012-06-14 05:38:40_real',
#                  '2014-09-01 10:37:27_real', '2014-09-01 17:13:18_real', '2014-08-11 11:35:13_real',
#                  '2014-09-03 06:37:24_real', '2014-09-03 19:37:15_real', '2014-08-15 08:48:43_real',
#                  '2012-09-02 17:42:41_real', '2013-06-25 19:25:21_real', '2014-09-10 12:09:48_real',
#                  '2014-09-11 01:07:17_real', '2014-08-17 23:34:43_real', '2014-08-18 01:13:49_real'
#                  ]

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_paths, content_path,
                               content_layers=CONTENT_LAYER_DEFAULT, style_layers=STYLE_LAYER_DEFAULT):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    content_img = image_loader(content_path)
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
            style_loss = AvgStyleLoss(style_paths, model)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], AvgStyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img, optimizer_choice):
    print('Optimizer: ', optimizer_choice)
    if optimizer_choice == 'lbfgs':
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif optimizer_choice == 'adam':
        optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)  # You can adjust the learning rate
    else:
        raise ValueError("Unsupported optimizer choice", optimizer_choice)
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std, content_path, style_paths,
                       input_img, num_steps=300, style_weight=1000000, content_weight=1, 
                       content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                       tv_weight=0.0, optimizer_choice='lbfgs', loss_choice='generic'):
    print('Building the style transfer model..')
    # print ("Content Layers: ", content_layers, "Style Layers: ", style_layers, "TV Weight: ", tv_weight, "Loss Choice: ", loss_choice)
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_paths, content_path,
                                                                     content_layers=content_layers, style_layers=style_layers)

    # we want to optimize the input and not the model parameters
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img, optimizer_choice=optimizer_choice)

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

            # loss = style_score + content_score
            # Add total variation regularization
            tv_loss = torch.sum(torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:])) + \
                        torch.sum(torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :]))
            tv_loss *= tv_weight

            if loss_choice == 'generic':
                if tv_weight != 0.0:
                    loss = style_score + content_score + tv_loss  # Include TV regularization in the loss
                else:
                    loss = style_score + content_score
            elif loss_choice == 'perceptual':
                # # Calculate the perceptual loss (VGG-style loss)
                # perceptual_loss = F.mse_loss(model(input_img), model(style_img))
                # perceptual_loss *= style_weight  # Adjust the weight as needed

                # loss = perceptual_loss + content_score  # Combine perceptual and content losses
                raise NotImplementedError() # TODO: need to make it work with avg style loss
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


def train(cnn, style_img_paths, content_img_path, output_img_path, metric_path, num_steps=300, style_weight=1000000, content_weight=1, 
          content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
          tv_weight=0.0, optimizer_choice='lbfgs', loss_choice='generic'):
    #
    # assert style_img.size() == content_img.size(), \
    #     "we need to import style and content images of the same size"

    plt.ion()
    input_img = image_loader(content_img_path).clone()
    output, style_loss, content_loss  = run_style_transfer(cnn, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD,
                                content_img_path, style_img_paths, input_img, num_steps, style_weight, content_weight, 
                                content_layers, style_layers, tv_weight, optimizer_choice, loss_choice)
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

def load_data(dataset, mode): 

    if dataset == "monet":
        DATASET_PATH = "./datasets/midterm_report/"
        style_folder, content_folder = os.path.join(DATASET_PATH, 'style_images'), os.path.join(DATASET_PATH, 'content_images')
        # load all file in folder
        style_img_paths = sorted(glob.glob(style_folder + "/*.jpg"))
        content_img_paths = sorted(glob.glob(content_folder + "/*.png"))
        img_out_dir, metric_out_dir = os.path.join(IMG_OUT_DIR, "monet"+ "_"+ mode),  os.path.join(METRIC_OUT_DIR, "monet"+ "_"+ mode )       

       
    elif dataset == "cityscape":
        # style_folder, content_folder = os.path.join('./datasets/midterm_report/', 'style_images'), os.path.join('./datasets/cityscapes/testA')
        style_folder, content_folder = os.path.join('./datasets/cityscapes/testA'), os.path.join('./datasets/cityscapes/testB')
        style_img_paths = sorted(glob.glob(style_folder + "/*.jpg"))
        content_img_paths = sorted(glob.glob(content_folder + "/*.jpg"))
        img_out_dir, metric_out_dir = os.path.join(IMG_OUT_DIR, "cityscape_mask_to_image"+ "_"+ mode), os.path.join(METRIC_OUT_DIR, "cityscape_mask_to_image"+ "_"+ mode)    
    else: 
        raise ValueError("Unsupported dataset choice")
    
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(metric_out_dir, exist_ok=True)

    return style_img_paths, content_img_paths, img_out_dir, metric_out_dir 

def get_cfg(mode): 
    if mode == "modified":
        cfg = mod_config
    elif mode == "original":
        cfg = ori_config
    else:
        raise ValueError("Unsupported mode choice")
    return cfg

if __name__ == '__main__':
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_weight", type=float, default=1000000)
    parser.add_argument("--content_weight", type=float, default=1)
    parser.add_argument("--dataset", type=str, default="monet") #dataset = monet or cityscape
    parser.add_argument("--mode", type=str, default="modified") #optimizer = lbfgs or adam

    args = parser.parse_args()
    print(args)
    
    style_img_paths, content_img_paths, img_out_dir, metric_out_dir = load_data(args.dataset, mode=args.mode)
    print ("Style Image: ", len(style_img_paths), "Content Image: ", len(content_img_paths))  
    
    cfg = get_cfg(args.mode)

    for idx, content_img_path in enumerate(content_img_paths):
        
        content_image_name = content_img_path.split("/")[-1].split(".")[0]

        # if content_image_name in OUTLIER_FILES:
        output_img_path = os.path.join(img_out_dir,  "avg"+"_"+content_image_name+".jpg")
        metric_path =  os.path.join(metric_out_dir, "avg" + "_"+ content_image_name+".json")

        print( "Content Image: ", content_image_name)

        train(cnn, style_img_paths, content_img_path, output_img_path, metric_path, num_steps=cfg.num_steps, style_weight=cfg.style_weight, content_weight=cfg.content_weight, 
              content_layers=cfg.content_layers, style_layers=cfg.style_layers,
            tv_weight=cfg.tv_weight, optimizer_choice=cfg.optimizer_choice, loss_choice=cfg.loss_choice)








