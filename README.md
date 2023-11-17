# How to download data
```bash
./download_cycle_gan_dataset.sh --dataset {d}

```
# How to train
- Specify the DATASET_PATH in avg_style_transfer.py or style_transfer.py. It is expected that the dataset contains 2 subfolders:
  - style_images
  - content_images
- Run the following command
```
# will do style transfer with avg gram matrix on the whole style dataset
python3 avg_style_transfer.py
```
- The output images and metrics (of each image) will be saved in the data/images and data/metrics

# How to collect metrics
Specify DATA_FOLDER, CONTENT_FOLDER and METRICS_FOLDER in compute_metrics.py

```bash
cd metrics
python3 compute_metrics.py
```