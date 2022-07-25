from icevision.all import *


def select_model(selection: int = 0, image_size: int = None):
    extra_args = {}
    model_type = None
    backbone = None

    if selection == 0:
        model_type = models.mmdet.vfnet
        backbone = model_type.backbones.resnet50_fpn_mstrain_2x

    if selection == 1:
        model_type = models.mmdet.retinanet
        backbone = model_type.backbones.resnet50_fpn_1x
        # extra_args['cfg_options'] = {
        #   'model.bbox_head.loss_bbox.loss_weight': 2,
        #   'model.bbox_head.loss_cls.loss_weight': 0.8,
        #    }

    if selection == 2:
        model_type = models.mmdet.faster_rcnn
        backbone = model_type.backbones.resnet101_fpn_2x
        # extra_args['cfg_options'] = {
        #   'model.roi_head.bbox_head.loss_bbox.loss_weight': 2,
        #   'model.roi_head.bbox_head.loss_cls.loss_weight': 0.8,
        #    }

    if selection == 3:
        model_type = models.mmdet.ssd
        backbone = model_type.backbones.ssd300

    if selection == 4:
        model_type = models.mmdet.yolox
        backbone = model_type.backbones.yolox_s_8x8

    if selection == 5:
        model_type = models.mmdet.yolof
        backbone = model_type.backbones.yolof_r50_c5_8x8_1x_coco

    if selection == 6:
        model_type = models.mmdet.detr
        backbone = model_type.backbones.r50_8x2_150e_coco

    if selection == 7:
        model_type = models.mmdet.deformable_detr
        backbone = model_type.backbones.twostage_refine_r50_16x2_50e_coco

    if selection == 8:
        model_type = models.mmdet.fsaf
        backbone = model_type.backbones.x101_64x4d_fpn_1x_coco

    if selection == 9:
        model_type = models.mmdet.sabl
        backbone = model_type.backbones.r101_fpn_gn_2x_ms_640_800_coco

    if selection == 10:
        model_type = models.mmdet.centripetalnet
        backbone = model_type.backbones.hourglass104_mstest_16x6_210e_coco

    elif selection == 11:
        # The Retinanet model is also implemented in the torchvision library
        model_type = models.torchvision.retinanet
        backbone = model_type.backbones.resnet50_fpn

    elif selection == 12:
        model_type = models.ross.efficientdet
        backbone = model_type.backbones.tf_lite0
        # The efficientdet model requires an img_size parameter
        extra_args['img_size'] = image_size

    elif selection == 13:
        model_type = models.ultralytics.yolov5
        backbone = model_type.backbones.small
        # The yolov5 model requires an img_size parameter
        extra_args['img_size'] = image_size

    return model_type, backbone, extra_args


def get_model_spec(backbone):
    if 'config_path' in backbone.__dict__.keys():
        model_spec = str(backbone.__dict__['config_path']).split('/')[-1].split('.')[0]
    else:
        model_spec = backbone.__dict__['model_name']

    print(model_spec)

    return model_spec
