from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.dropout import Dropout2d
import math
from torch.nn.modules.linear import Linear
import pdb
try:
    from timm.models import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, \
    tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, \
    xception

    from efficientnet_pytorch.model import EfficientNet
except:
    from timm.models import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, \
    tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, \
    xception

    from efficientnet_pytorch.model import EfficientNet


__all__ = ['BinaryClassifier']


encoder_params = {
    "xception": {
        "features": 2048,
        "init_op": partial(xception, pretrained=True)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=True)
    },
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True)
    },
    "efficientnet-b4": {
        "features": 1792,
        "init_op": partial(EfficientNet.from_pretrained, model_name='efficientnet-b4')
    },
    "efficientnet-b5": {
        "features": 2048,
        "init_op": partial(EfficientNet.from_pretrained, model_name='efficientnet-b5')
    }
}



class BinaryClassifier(nn.Module):
    def __init__(self, encoder, num_classes=1, drop_rate=0.2, has_feature=True,feature_dim=128,**kwargs) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"](**kwargs)
        self.global_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(drop_rate)
        self.channel_drop = Dropout2d(drop_rate)
        self.has_feature = has_feature
        self.feature = Linear(encoder_params[encoder]["features"], feature_dim)
        self.fc = Linear(1792, num_classes)

    def forward(self, x):
        featuremap = self.encoder.forward_features(x)
        x = self.global_pool(featuremap).flatten(1)
        output = self.fc(x)
        if self.has_feature:
            return output,featuremap
        return featuremap



if __name__ == '__main__':
    model = BinaryClassifier("efficientnet-b5")
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        input = torch.rand(4, 3, 320, 320)
        input = input.cuda()
        out = model(input)
        print(out.shape)
    