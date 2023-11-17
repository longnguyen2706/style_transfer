import glob
import json
import os

import numpy as np

from eval_metrics import EvaluationMetrics

DATA_FOLDER = "./datasets/midterm_report/"
CONTENT_FOLDER = os.path.join(DATA_FOLDER, 'content_images')
METRICS_FOLDER = "./data/metrics/"

# cal metrics on styled image and save to file
def cal_image_metrics():
    eval_metrics = EvaluationMetrics(device='cuda')
    content_folder = CONTENT_FOLDER
    # load all file in folder
    content_img_paths = sorted(glob.glob(content_folder + "/*.png"))

    for content_img_path in content_img_paths:
        content_image_name = content_img_path.split("/")[-1].split(".")[0]

        generated_img_path = "./data/images/" + "avg" + "_" + content_image_name + ".jpg"

        print("Content Image: ", content_image_name)

        # Compute SSIM
        ssim_score = eval_metrics.compute_ssim(content_img_path, generated_img_path)
        print(f'SSIM: {ssim_score}')

        # Compute PSNR
        psnr_score = eval_metrics.compute_psnr(content_img_path, generated_img_path)
        print(f'PSNR: {psnr_score}')

        # Preprocess the images for feature-based similarity and LPIPS
        content_tensor = eval_metrics.preprocess_image(content_img_path)
        generated_tensor = eval_metrics.preprocess_image(generated_img_path)

        # Compute feature-based similarity
        feature_similarity = eval_metrics.compute_feature_similarity(content_tensor, generated_tensor)
        print(f'Feature-based similarity (cosine): {feature_similarity}')

        # Compute LPIPS and ArtFID
        lpips_score, art_fid_score = eval_metrics.compute_lpips_and_artFID(content_tensor, generated_tensor)
        print(f'LPIPS: {lpips_score}')
        print(f'ArtFID:{art_fid_score}')

        metrics_path = "./data/metrics/" + "avg2" + "_" + content_image_name + ".json"
        with open(metrics_path, 'w') as f:
            data = {}
            data["ssim"] = ssim_score
            data["psnr"] = psnr_score
            data["feature_similarity"] = feature_similarity
            data["lpips"] = lpips_score
            data["art_fid"] = art_fid_score
            f.write(json.dumps(data, indent=4))


def cal_mean(arr):
    # print (arr, len(arr))
    return np.sum(arr) / len(arr)


def cal_std(arr):
    mean = cal_mean(arr)
    return (sum([(x - mean) ** 2 for x in arr]) / len(arr)) ** 0.5


def cal_dataset_metrics():
    style_loss_scores, content_loss_scores, ssim_scores, psnr_scores, \
        feature_similarity_scores, lpips_scores, art_fid_scores = [], [], [], [], [], [], []

    content_img_paths = sorted(glob.glob(CONTENT_FOLDER + "/*.png"))

    for content_img_path in content_img_paths:
        content_image_name = content_img_path.split("/")[-1].split(".")[0]
        metrics1_path = METRICS_FOLDER + "avg" + "_" + content_image_name + ".json"
        metrics2_path = METRICS_FOLDER + "avg2" + "_" + content_image_name + ".json"
        with open(metrics1_path, 'r') as f:
            metrics1 = json.load(f)
        with open(metrics2_path, 'r') as f:
            metrics2 = json.load(f)

        style_loss_scores.append(metrics1["style_loss"])
        content_loss_scores.append(metrics1["content_loss"])
        ssim_scores.append(metrics2["ssim"])
        psnr_scores.append(metrics2["psnr"])
        feature_similarity_scores.append(metrics2["feature_similarity"])
        lpips_scores.append(metrics2["lpips"])
        art_fid_scores.append(metrics2["art_fid"])

    # ignore the outlier
    ignore_idx1 = [idx for idx, score in enumerate(style_loss_scores) if score > 100]
    ignore_idx2 = [idx for idx, score in enumerate(content_loss_scores) if score > 100]
    # merge 2
    ignore_idx = list(set(ignore_idx1 + ignore_idx2))
    ignored_images = [content_img_paths[idx].split("/")[-1].split(".")[0] for idx in ignore_idx]
    print("Ignore index: ", ignore_idx)
    print("Ignore images: ", ignored_images)

    style_loss_scores = [score for idx, score in enumerate(style_loss_scores) if idx not in ignore_idx]
    content_loss_scores = [score for idx, score in enumerate(content_loss_scores) if idx not in ignore_idx]
    ssim_scores = [score for idx, score in enumerate(ssim_scores) if idx not in ignore_idx]
    psnr_scores = [score for idx, score in enumerate(psnr_scores) if idx not in ignore_idx]
    feature_similarity_scores = [score for idx, score in enumerate(feature_similarity_scores) if idx not in ignore_idx]
    lpips_scores = [score for idx, score in enumerate(lpips_scores) if idx not in ignore_idx]
    art_fid_scores = [score for idx, score in enumerate(art_fid_scores) if idx not in ignore_idx]
    print (len(art_fid_scores), art_fid_scores)

    # cal and print mean and std print with 2 decimal places
    print('Style Loss mean and std: {:.4f} {:.4f}'.format(cal_mean(style_loss_scores), cal_std(style_loss_scores)))
    print("Content Loss mean and std: {:.4f} {:.4f}".format(cal_mean(content_loss_scores), cal_std(content_loss_scores)))
    print ("SSIM mean and std: {:.4f} {:.4f}".format(cal_mean(ssim_scores), cal_std(ssim_scores)))
    print ("PSNR mean and std: {:.4f} {:.4f}".format(cal_mean(psnr_scores), cal_std(psnr_scores)))
    print ("Feature Similarity mean and std: {:.4f} {:.4f}".format(cal_mean(feature_similarity_scores), cal_std(feature_similarity_scores)))
    print ("LPIPS mean and std: {:.4f} {:.4f}".format(cal_mean(lpips_scores), cal_std(lpips_scores)))
    print ("ArtFID mean and std: {:.4f} {:.4f}".format(cal_mean(art_fid_scores), cal_std(art_fid_scores)))


if __name__ == "__main__":
    # cal_image_metrics()
    cal_dataset_metrics()
