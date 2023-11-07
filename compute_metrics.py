import glob
import json
import os

from eval_metrics import EvaluationMetrics

if __name__ == "__main__":
    eval_metrics = EvaluationMetrics(device='cuda')

    DATA_FOLDER = "./datasets/midterm_report/"
    style_folder, content_folder = os.path.join(DATA_FOLDER, 'style_images'), os.path.join(DATA_FOLDER, 'content_images')
    # load all file in folder
    style_img_paths = sorted(glob.glob(style_folder + "/*.jpg"))
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
