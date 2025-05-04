import torch.nn as nn
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_instance_segmentation_model(num_classes: int = 5, pretrained_backbone=True):
    """Return an enhanced Mask R-CNN with the correct #classes.

    Parameters
    ----------
    num_classes : int
        Number of classes including background (for 4 cell types => 5)
    pretrained_backbone : bool
        Whether to use pretrained backbone
    """

    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
        trainable_layers=3,  # layer3, layer4, FPN
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        box_detections_per_img=300,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        min_size=512,
        rpn_score_thresh=0.05,
        rpn_nms_thresh=0.7,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
    )

    # Replace the box predictor with one that includes more features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Enhance the mask predictor with more capacity
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512  # Increased from 256
    model.roi_heads.mask_predictor = EnhancedMaskPredictor(
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


class EnhancedMaskPredictor(nn.Module):
    """Enhanced mask predictor with extra capacity for fine details."""

    def __init__(self, in_channels, hidden_layer, num_classes):
        super(EnhancedMaskPredictor, self).__init__()

        # More capacity for detailed feature extraction
        self.conv1 = nn.Conv2d(in_channels, hidden_layer, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_layer)
        self.conv2 = nn.Conv2d(hidden_layer, hidden_layer, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_layer)
        self.conv3 = nn.Conv2d(hidden_layer, hidden_layer, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_layer)
        self.conv4 = nn.Conv2d(hidden_layer, hidden_layer, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_layer)

        # Final layer for mask prediction
        self.conv5_mask = nn.Conv2d(hidden_layer, num_classes, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return self.conv5_mask(x)
