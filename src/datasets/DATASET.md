## Datasets & Data Preparation
In the process of fine-tuning/linear probing the embedding models, the following dataset can be used:

| Dataset                                                                                                           |        Domain         |    Config key    | Note                                                                                                    |
|-------------------------------------------------------------------------------------------------------------------|:---------------------:|:----------------:|---------------------------------------------------------------------------------------------------------|
| [Products-10k](https://products-10k.github.io)                                                                    |    Packaged goods     |  `products_10k`  |                                                                                                         |
 | [Google Landmarks v2](https://www.kaggle.com/c/landmark-recognition-2021/data)                                    |       Landmarks       |     `gldv2`      | cleaned subset of GLDv2 is used.                                                                        |
| [DeepFashion (Consumer to Shop)](https://www.kaggle.com/datasets/sangamman/deepfashion-consumer-to-shop-training) | Apparel & Accessories |  `deep_fashion`  |                                                                                                         |
| [MET Artwork](http://cmp.felk.cvut.cz/met/)                                                                       |        Artwork        |    `met_art`     |                                                                                                         |
 | [Shopee](https://www.kaggle.com/competitions/shopee-product-matching/data)                                        |    Packaged goods     |     `shopee`     |                                                                                                         |
 | [H&M Personalized Fashion](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) | Apparel & Accessories |       `hm`       |                                                                                                         |
 | [RP2K](https://www.pinlandata.com/rp2k_dataset/)                                                                  |    Packaged goods     |      `rp2k`      |                                                                                                         |
 | [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)                                     |    Packaged goods     |      `sop`       |                                                                                                         |
 | [Fashion200k](https://github.com/xthan/fashion-200k)                                                              | Apparel & Accessories |  `fashion200k`   | annotations in csv format (see `data/fashion200k_train.csv`)                                            |
 | [Food Recognition 2022](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022#datasets)              |     Food & Dishes     |   `food_rec22`   | dataset must to bee preprocessed according to `data/food_rec22_preprocess.py`                           |
 | [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)                              |         Cars          | `stanford_cars`  | annotations in csv format [here](https://github.com/BotechEngineering/StanfordCarsDatasetCSV/tree/main) |
 | [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)                                                   | Apparel & Accessories | `deep_fashion2`  | dataset must to bee preprocessed according to `data/deep_fashion2_preprocess.py`                        |
 | [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)                                            |     Food & Dishes     |    `food101`     | test images are used for training.                                                                      |
 | [Furniture 180](https://www.kaggle.com/datasets/andreybeyn/qudata-gembed-furniture-180)                           |       Furniture       |  `furniture180`  | annotations in csv format (see `data/furniture180_train.csv`)                                           |
 | [Storefornts 146](https://www.kaggle.com/datasets/kerrit/storefront-146)                                          |      Storefronts      | `storefronts146` | annotations in csv format (see `data/storefronts146_train.csv`)                                         |

Download the datasets and place them in a `<data_dir>` of your choice. The directory structure should look as follows:
```
<data_dir>/
├── m4d-35k_train.csv
├── products-10k/
│   ├── train
│   └── train.csv
├── google_landmark_recognition_2021/
│   ├── train
│   └── train.csv
├── deepfashion/
│   ├── train
│   └── deepfashion_train.json
├── met_dataset/
│   ├── MET
│   └── ground_truth/MET_database.json
├── shopee/
│   ├── train_imahes
│   └── train.csv
├── hm_personalized_fashion/
│   ├── images
│   └── articles.csv
├── rp2k/
│   ├── train
│   └── train.csv
├── stanford_online_products/
│   ├── <img_dirs>
│   └── Ebay_train.txt
├── fashion200k/
│   ├── women
│   └── fashion200k_train.csv
├── fr22_train_v2/
│   ├── images
│   ├── preprocessed_imgs
│   ├── annotations.json
│   └── train.csv
├── stanford_cars/
│   ├── cars_train
│   └── sc_train.csv
├── deep_fashion2/
│   ├── image
│   ├── annos
│   ├── preprocessed_imgs
│   └── train.csv
├── food-101/
│   ├──images
│   └── meta/test.json
├── furniture_180/
│   ├── <img_dirs>
│   └── furniture180_train.csv
└── storefronts_146/
    ├── <img_dirs>
    └── storefronts146_train.csv
```

The following parameters in the configuration file in `configs/` can be adjusted regarding the training data and data loading:
- `DATASET.names`: list of dataset names to be used for training. The dataset names are the `config keys` from the table above.
- `DATALOADER.batch_size`: batch size for training data loading.
- `DATALOADER.num_workers`: number of workers for data loading.
- `TRANSFORM.name`: name of the transformation to be used for data augmentation, supported are the training transforms 
from [CLIP](https://github.com/openai/CLIP) (`openai-clip`), [OpenCLIP](https://github.com/mlfoundations/open_clip) 
(`openclip`), and [SigLIP](https://arxiv.org/abs/2303.15343) (`siglip`).

Image size and normalization (mean and std from the pre-training dataset) are automatically determined from the pre-trained 
foundation model used.