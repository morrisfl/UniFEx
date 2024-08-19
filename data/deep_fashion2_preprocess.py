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


def data_preprocessing(root):
    """Preprocess the DeepFashion2 dataset as follows:
            1. Crop the images with 2% offset from the bounding box. If an image contains multiple bounding boxes, crop
            the image for each bounding box.
            2. Create a csv file containing the cropped image paths and their corresponding labels.
    """
    img_dir = os.path.join(root, "image")
    ann_dir = os.path.join(root, "annos")
    output_dir = os.path.join(root, "preprocessed_imgs")

    with open(os.path.join(root, "train.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "label"])

        for img in tqdm(os.listdir(img_dir)):
            if img == ".DS_Store":
                continue

            img_id = img.split(".")[0]
            ann_path = os.path.join(ann_dir, img_id + ".json")

            with open(ann_path, "r") as j:
                ann = json.load(j)

            keys = list(ann.keys())
            pair_id = ann["pair_id"]
            for key in keys:
                if key in ["source", "pair_id"]:
                    continue

                item = ann[key]
                style = int(item["style"])
                if style > 0:
                    cat_name = item["category_name"]
                    bbox = item["bounding_box"]
                    label = f"{pair_id}_{style}"

                    try:
                        image = Image.open(os.path.join(img_dir, img))
                    except:
                        print(f"Erorr opening image: {os.path.join(img_dir, img)}")
                        continue

                    if not os.path.exists(os.path.join(output_dir, cat_name, label)):
                        os.makedirs(os.path.join(output_dir, cat_name, label))

                    if len(keys) > 3:
                        crop_img = crop_image(image, bbox)
                        crop_img.save(os.path.join(output_dir, cat_name, label, img))
                    else:
                        image.save(os.path.join(output_dir, cat_name, label, img))

                    writer.writerow([os.path.join("preprocessed_imgs", cat_name, label, img), label])

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
    data_root = args.data_root

    data_preprocessing(data_root)
