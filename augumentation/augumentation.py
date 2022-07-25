from icevision.all import *
from albumentations.augmentations.crops.transforms import CropAndPad
from albumentations.augmentations.transforms import *
import albumentations as A


def medical_aug(image_size, presize_multi):
    train_tfms = tfms.A.Adapter([
        *tfms.A.aug_tfms(size=image_size, presize=int(image_size * presize_multi), ),
        A.Normalize(),
        A.HorizontalFlip(p=0.5),
        A.Flip(p=0.5),
        A.Blur(blur_limit=5, p=0.5),
        A.MedianBlur(blur_limit=5, p=0.5),
        A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0.5, always_apply=False, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Cutout(num_holes=4, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3, val_shift_limit=0.3, p=0.5),
        # A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.5)
    ])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
    return train_tfms, valid_tfms


def train_valid_aug(image_size, presize_multi):
    # Transforms
    train_tfms = tfms.A.Adapter([
        *tfms.A.aug_tfms(size=image_size, presize=int(image_size * presize_multi), ),
        tfms.A.Normalize()
    ])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

    return train_tfms, valid_tfms


class ImageAugmentation:
    def __init__(self, medical_augmentation: bool = True,
                 my_image_size: int = 256, presize_multi: float = 1.5,
                 model_spec: str = None, bs_train: int = 32, bs_val: int = 32):
        self.medical_augmentation = medical_augmentation
        self.my_image_size = my_image_size
        self.presize_multi = presize_multi
        self.model_spec = model_spec
        self.bs_train = bs_train
        self.bs_val = bs_val

        self.model_name, self.model_dir = self.create_model_name_and_dir()
        self.train_tfms, self.valid_tfms = self.get_augmentation()

    def create_model_name_and_dir(self):
        if self.medical_augmentation:
            model_name = f'{self.model_spec}-bst-{self.bs_train}-bsv-{self.bs_val}-medical-val-200'
            model_dir = f'/home/fabio/Documents/segmentation/models/bbox_{model_name}'
        else:
            model_name = f'{self.model_spec}-bst-{self.bs_train}-bsv-{self.bs_val}-standard_aug'
            model_dir = f'/home/fabio/Documents/segmentation/models/bbox_{model_name}'

        return model_name, model_dir

    def get_augmentation(self):
        if self.medical_augmentation:
            train_tfms, valid_tfms = medical_aug(self.my_image_size, self.presize_multi)
            print('medical_augmentation')
        else:
            train_tfms, valid_tfms = train_valid_aug(self.my_image_size, self.presize_multi)
            print('Standard_augmentation')

        return train_tfms, valid_tfms


class OrganDataLoaders:
    def __init__(self, records, image_augmentation, negative_samples):
        self.records = records
        self.image_augmentation = image_augmentation
        self.negative_samples = negative_samples
        self.train_ds, self.valid_ds = self.create_dls()

    def create_dls(self):
        if self.negative_samples:

            train_ds = Dataset(self.records.train_rec + self.records.negative_train_rec, self.image_augmentation.train_tfms)
            valid_ds = Dataset(self.records.valid_rec + self.records.negative_valid_rec, self.image_augmentation.valid_tfms)
            print('negative sample')
        else:
            train_ds = Dataset(self.records.train_rec, self.image_augmentation.train_tfms)
            valid_ds = Dataset(self.records.valid_rec, self.image_augmentation.valid_tfms)
            print('samples')
        print(len(train_ds), len(valid_ds))

        return train_ds, valid_ds

