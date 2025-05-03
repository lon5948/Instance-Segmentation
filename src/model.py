import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes: int = 5, pretrained_backbone=True):
    """Return a Mask R-CNN with the correct #classes.
    num_classes includes background → for 4 cell types => 5.
    """
    model = maskrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained_backbone else None,
        box_detections_per_img=100,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=500,
        min_size=600,
    )

    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # Initialize weights for new layers
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Apply only to the replaced modules
    model.roi_heads.box_predictor.apply(init_weights)
    model.roi_heads.mask_predictor.apply(init_weights)

    return model
