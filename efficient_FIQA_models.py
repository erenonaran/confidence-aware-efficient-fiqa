import torch
import torch.nn as nn
import timm
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Swin_b_IQA(nn.Module):
    def __init__(self, is_pretrained=False):
        super(Swin_b_IQA, self).__init__()

        if is_pretrained:
            model = models.swin_b(weights='Swin_B_Weights.DEFAULT')
        else:
            model = models.swin_b()

        model.head = Identity()
        self.feature_extraction = model

        self.quality = self.quality_regression(1024, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x):

        x = self.feature_extraction(x)
        x = self.quality(x)

        return x


class FIQA_Swin_B(torch.nn.Module):

    def __init__(self, pretrained_path, is_pretrained=False):

        super(FIQA_Swin_B, self).__init__()

        swin_b = Swin_b_IQA(is_pretrained)
        if pretrained_path!=None:
            print('load overall model')
            state_dict = torch.load(pretrained_path)
            state_dict = remove_prefix(state_dict, "module.")
            state_dict = remove_prefix(state_dict, "feature_extraction.")

            swin_b.load_state_dict(state_dict)
        swin_b.quality = Identity()

        self.feature_extraction = swin_b

        self.quality = self.quality_regression(1024, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x, return_feat=False):
        feat = self.feature_extraction(x)   # [B, 1024]
        out  = self.quality(feat)           # [B, 1]

        if return_feat:
            return out, feat
        return out

class FIQA_EdgeNeXt_XXS(nn.Module):
    def __init__(self, pretrained_path=None, is_pretrained=True):
        super().__init__()

        backbone = timm.create_model('edgenext_xx_small', pretrained=is_pretrained)
        backbone.head = Identity()

        self.feature_extraction = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quality = nn.Sequential(
            nn.Linear(168, 128),
            nn.Linear(128, 1),
        )

        if pretrained_path is not None:
            print("Loading FIQA checkpoint:", pretrained_path)
            ckpt = torch.load(pretrained_path, map_location="cpu")

            # handle both formats
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            else:
                state = ckpt

            self.load_state_dict(state, strict=True)



    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x, return_feat=False):
        feat_map = self.feature_extraction(x)     # [B, 168, H, W] (likely)
        feat_map = self.avg_pool(feat_map)        # [B, 168, 1, 1]
        feat = torch.flatten(feat_map, 1)         # [B, 168]
        out = self.quality(feat)                  # [B, 1]

        if return_feat:
            return out, feat
        return out


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = 
