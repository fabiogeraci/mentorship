import os
import gc
import random
import sys
sys.path.append('/home/fabio/Documents/mentorship')

import pandas as pd
import numpy as np
import wandb

from icevision.all import *
from fastai.callback.wandb import *
from fastai.callback.all import *

from icevision.data.data_splitter import SingleSplitSplitter
from icevision.metrics.coco_metric.coco_metric import COCOMetric, COCOMetricType

import parsers
from architectures.archs_backbones import select_model, get_model_spec
from augumentation.augumentation import ImageAugmentation, OrganDataLoaders

from parsers.samples_parsers import ProstateParser, bbox_record
from parsers.negative_samples_parsers import NegativeSamplesParser, negative_sample_record

import warnings
warnings.filterwarnings('ignore')

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_colwidth', 150)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


class Dataset:

    def __init__(self, data_path):
        self.data_path: str = data_path
        self.df_train_set = None
        self.df_train_neg = None
        self.df_val_set = None
        self.df_val_neg = None
        self.get_train()
        self.get_valid()

    def get_train(self):
        # Training DataFrame
        self.df_train_set = pd.read_csv(os.path.join(self.data_path, 'df_train_set.csv'))
        # Negative Samples Training DataFrame
        self.df_train_neg = pd.read_csv(os.path.join(self.data_path, 'df_train_neg.csv'))

    def get_valid(self):
        # Validation DataFrame
        self.df_val_set = pd.read_csv(os.path.join(self.data_path, 'df_val_set.csv'))
        # Negative Samples Validation DataFrame
        self.df_val_neg = pd.read_csv(os.path.join(self.data_path, 'df_val_neg.csv'))


class Records(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_rec = None
        self.valid_rec = None
        self.negative_train_rec = None
        self.negative_valid_rec = None
        self.class_map_ = None
        self.generate_records()

    def generate_records(self):
        self.generate_positive_records()
        self.generate_negative_records()

    def generate_positive_records(self):
        parser_prostate = ProstateParser(bbox_record, self.df_train_set, os.path.join(self.data_path, 't2_axial'))
        self.train_rec = parser_prostate.parse(SingleSplitSplitter(), autofix=False)[0]
        parser_prostate = ProstateParser(bbox_record, self.df_val_set, os.path.join(self.data_path, 't2_axial'))
        self.valid_rec = parser_prostate.parse(SingleSplitSplitter(), autofix=False)[0]
        self.class_map_ = parser_prostate.class_map

    def generate_negative_records(self):
        negative_samples = NegativeSamplesParser(negative_sample_record, self.df_train_neg, os.path.join(self.data_path, 't2_axial'))
        self.negative_train_rec = negative_samples.parse(SingleSplitSplitter(), autofix=False)[0]
        negative_samples = NegativeSamplesParser(negative_sample_record, self.df_val_neg, os.path.join(self.data_path, 't2_axial'))
        self.negative_valid_rec = negative_samples.parse(SingleSplitSplitter(), autofix=False)[0]


if __name__ == "__main__":

    main_path = os.path.abspath(sys.argv[0] + "/../..")
    DATA_PATH = os.path.join(main_path, 'data')

    records = Records(DATA_PATH)

    medical_augmentation = True
    negative_samples = True
    my_image_size = 256
    presize_multi = 1.5
    bs_train = 32
    bs_val = 32

    model_type, backbone, extra_args = select_model(selection=0, image_size=my_image_size)

    model_spec = get_model_spec(backbone)

    image_augmentation = ImageAugmentation(medical_augmentation,
                                           my_image_size, presize_multi, model_spec,
                                           bs_train, bs_val)
    # WANDB_API_KEY = os.environ["WANDB_API_KEY"]
    # wandb.login(key=WANDB_API_KEY, relogin=True)
    # wandb.init(project="mentorship_seg", name=image_augmentation.model_name, entity="mentorship", reinit=True)

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox, print_summary=False)]

    WandbCallback_ = WandbCallback()
    SaveModelCallback_ = SaveModelCallback()
    ReduceLROnPlateau_ = ReduceLROnPlateau(monitor="COCOMetric")

    cbs = [
        WandbCallback_,
        SaveModelCallback_,
        # ReduceLROnPlateau_,
    ]

    # Instantiate the model
    model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(records.class_map_), **extra_args)

    # Data Loaders
    dataloaders_ = OrganDataLoaders(records, image_augmentation, negative_samples)
    train_dl = model_type.train_dl(dataloaders_.train_ds, batch_size=bs_train, shuffle=True)
    valid_dl = model_type.valid_dl(dataloaders_.valid_ds, batch_size=bs_val, shuffle=False)

    # Initialize Learner
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learn = model_type.fastai.learner(dls=[train_dl, valid_dl],
                                      model=model,
                                      metrics=metrics,
                                      cbs=cbs,
                                      model_dir=image_augmentation.model_dir,
                                      device=device
                                      )

    gc.collect()
    torch.cuda.empty_cache()

    seed = 2022
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

