from yacs.config import CfgNode as CN

_C = CN()
_C.SEED = 42  # Random seed for reproducibility.
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.name = 'siglip-sovit-400m_proj_layer64_arcface_adam2e-2_m4d-35k'  # Name of the model, used for logging and
# checkpointing. Gets created automatically when training.
_C.MODEL.embedding_dim = 64  # Dimensionality of the embeddings/feature vector.

# Save model options
_C.MODEL.output_dir = "results/"  # Directory where the model and logs will be saved.
_C.MODEL.cloud_upload = True  # Whether to upload the model to the cloud.

# backbone
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.output_dim = 1152  # Output embedding dimensionality of the backbone.
_C.MODEL.BACKBONE.freeze_backbone = True  # Whether to freeze the backbone.
_C.MODEL.BACKBONE.type = "siglip"  # Which foundation model type to use. Options are 'clip', 'clipav2', 'meta-clip',
# 'eva02', 'clip_convnext', 'dinov2', 'sam' or 'siglip'.
_C.MODEL.BACKBONE.network_arch = "sovit400m"  # Name of the network architecture to use. Used to generate name.
_C.MODEL.BACKBONE.model_name = "vit_so400m_patch14_siglip_384"  # name of the foundational model to use.
_C.MODEL.BACKBONE.weights = ""  # Name of the pretrained weights to use. Can be empty for timm models.
_C.MODEL.BACKBONE.proj_layer = False  # Whether to include the projection layer of the ViT

# neck
_C.MODEL.NECK = CN()
_C.MODEL.NECK.neck_type = "proj_layer"  # Type of neck to use. Options are 'proj_layer' or 'pooling'.
_C.MODEL.NECK.dropout = 0.2  # Dropout rate for the neck.

# head
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.name = "ArcFace"  # Name of the head to use. Options are 'ArcFace', 'DynM-ArcFace', 'AdaCos', 'LiArcFace',
# 'CurricularFace', and 'AdaFace'.
_C.MODEL.HEAD.k = 1  # Number of centers to use in the ArcFace head.
_C.MODEL.HEAD.s = 30.0  # Scale factor for the logits.
_C.MODEL.HEAD.dynamic_s = True  # Whether to use dynamic s calculation for AdaCos head.
_C.MODEL.HEAD.m = 0.5  # Margin for the logits.
_C.MODEL.HEAD.m_max = 0.5  # Max margin for DynM-ArcFace head.
_C.MODEL.HEAD.m_min = 0.2  # Min margin for the DynM-ArcFace head.
_C.MODEL.HEAD.h = 0.333  # Hyperparameter for the AdaFace head.
_C.MODEL.HEAD.t_alpha = 1.0  # Hyperparameter for the AdaFace head.

# -----------------------------------------------------------------------------
# Dataset & Dataloader settings
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root = "/data"  # Root directory of the datasets.
_C.DATASET.names = ["m4d-35k"]  # List of datasets names to include in the train_pipeline
_C.DATASET.cls_dist_file = "data/m4d-35k_samples_per_cls.csv"  # File with the number of samples per class of the
# training dataset. If the path does not exist, the file will be created. Is needed to calculate margin values for the
# DynM-ArcFace.

# DataLoader
_C.DATALOADER = CN()
_C.DATALOADER.batch_size = 128  # Batch size to use in the dataloader.
_C.DATALOADER.num_workers = 8  # Number of workers to use in the dataloader.

# Augmentation & Transformation
_C.TRANSFORM = CN()
_C.TRANSFORM.name = "openai-clip"  # Name of the transform to use. Options are 'clip_default'
_C.TRANSFORM.size = 384  # Resize size of the image.
_C.TRANSFORM.mean = (0.48145466, 0.4578275, 0.40821073)  # Mean of the datasets (gets updated automatically based used
# pre-trained ViT)
_C.TRANSFORM.std = (0.26862954, 0.26130258, 0.27577711)  # Standard deviation of the datasets (gets updated
# automatically based used pre-trained ViT).

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.epoch_based = True  # Whether to train using iterations or epochs.
_C.TRAIN.epochs = 10  # Number of epochs to train.
_C.TRAIN.iterations = 0  # Number of iterations to train. Get calculated if epoch_based is True.
_C.TRAIN.save_epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Epochs to save the model.
_C.TRAIN.save_iter = 2500  # Save the model every saver_iter iterations. Only active if epoch_based is False
# (get automatically calculated based on the batch size and seen images, 320,000 / 128 = 2,500).
_C.TRAIN.seen_img_save = 320000  # Save the model every seen_img_save seen images.
_C.TRAIN.print_freq = 500  # Print frequency of the train_pipeline logs.
_C.TRAIN.device = "cuda:0"  # Device to use for training.
_C.TRAIN.gpu_ids = [0, 1]  # GPU IDs to use for training.
_C.TRAIN.data_parallel = False  # Whether to use data parallelism.

# loss
_C.LOSS = CN()
_C.LOSS.name = "CrossEntropyLoss"  # Name of the loss to use.

# optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.name = "Adam"  # Name of the optimizer to use. Options are 'SGD', 'Adam', 'AdamW', 'SAM_AdamW', or
# 'SAM_AdamW'
_C.OPTIMIZER.lr = 2e-3  # Learning rate to use for the optimizer.
_C.OPTIMIZER.weight_decay = 1e-4  # Weight decay to use for the optimizer.
_C.OPTIMIZER.momentum = 0.9  # Momentum to use for the SGD optimizer.

# scheduler
_C.SCHEDULER = CN()
_C.SCHEDULER.epoch_based = False  # Whether to use epochs or iterations for the scheduler.
_C.SCHEDULER.name = "cosine"  # Name of the scheduler to use. Options are None, 'cosine'
_C.SCHEDULER.warmup = "linear"  # Type of warmup to use. Options are None, 'linear', and 'exponential'
_C.SCHEDULER.warmup_steps = 1  # Number of warmup steps to use (get automatically calculated based on the batch size
# and number of seen images for warmup) If the value is 1, the warmup steps are equivalent to the number of iterations
# of one epoch.
_C.SCHEDULER.seen_img_warmup = 160000  # Number of seen images to use for the warmup.
_C.SCHEDULER.min_lr = 1e-3  # Minimum learning rate to use for the CosineAnnealingLR

# -----------------------------------------------------------------------------
# Google Drive settings
# -----------------------------------------------------------------------------
_C.GOOGLE_DRIVE = CN()
_C.GOOGLE_DRIVE.api_name = "drive"  # Name of the api to use.
_C.GOOGLE_DRIVE.api_version = "v3"  # Version of the api to use.
_C.GOOGLE_DRIVE.scope = ["https://www.googleapis.com/auth/drive"]  # Scope of the api to use.
_C.GOOGLE_DRIVE.folder_id = ""  # Folder ID in GD where the model will be saved.
_C.GOOGLE_DRIVE.client_secret_path = "credentials/client_secret.json"  # Path to the client_secret.json file.
_C.GOOGLE_DRIVE.token_path = "credentials/token_drive_v3.pickle"  # Path to the token.pickle file.


def get_cfg_defaults():
    return _C.clone()


if __name__ == "__main__":
    cfg = get_cfg_defaults()

    sampling = cfg.DATASET.sampling_p10k
    print(type(sampling))
