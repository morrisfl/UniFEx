import argparse
import csv
import json
import os

from PIL import Image
from tqdm import tqdm


def pars_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="path to the data root directory.")

    return parser.parse_args()


def fr22_preprocessing(root):
    """Preprocess the Food Recognition 2022 dataset as follows:
            1. Crop the images with 2% offset from the bounding box. If an image contains multiple bounding boxes, crop
            the image for each bounding box.
            2. Create a csv file containing the cropped image paths and their corresponding labels.
    """
    img_dir = os.path.join(root, "images")
    ann_path = os.path.join(root, "annotations.json")
    out_dir = os.path.join(root, "preprocessed_imgs")

    with open(ann_path, "r") as j:
        ann_train = json.load(j)

    annotations = ann_train["annotations"]

    img_name_to_annotations = {}
    for image in tqdm(ann_train["images"], desc="Creating image name to annotations dict"):
        img_ann = [ann for ann in annotations if ann["image_id"] == image["id"]]
        img_name = image["file_name"]
        img_name_to_annotations[img_name] = img_ann

    with open(os.path.join(root, "train.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "label"])

        for name in tqdm(img_name_to_annotations, desc="Crop and save images"):
            ann = img_name_to_annotations[name]
            img_path = os.path.join(img_dir, name)
            image = Image.open(img_path)
            for item in ann:
                bbox = item["bbox"]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                label = item["category_id"]
                try:
                    crop_img = crop_image(image, bbox)

                    save_path = os.path.join(out_dir, str(label))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    crop_img.save(os.path.join(save_path, name))
                    writer.writerow([os.path.join("preprocessed_imgs", str(label), name), label])
                except Exception as e:
                    print(e)
                    print("Failed to crop image: ", name)


def crop_image(image, box):
    width_offset = (box[2] - box[0]) * 0.02
    height_offset = (box[3] - box[1]) * 0.02
    left = max(box[0] - width_offset, 0)
    top = max(box[1] - height_offset, 0)
    right = min(box[2] + width_offset, image.width)
    bottom = min(box[3] + height_offset, image.height)
    crop = image.crop((left, top, right, bottom))
    return crop


if __name__ == "__main__":
    args = pars_args()

    fr22_preprocessing(args.data_root)
