from torchvision import transforms

def make_aug_transforms(
    image_size=448,
    crop_size=448,
    max_rotation=5,
    translate_frac=0.04,
    scale_range=(0.97, 1.03),
    hflip_prob=0.5,
):
    """
    Alignment-aware augmentations for FIQA confidence estimation
    """

    return transforms.Compose([
        transforms.Resize(image_size),

        transforms.RandomAffine(
            degrees=max_rotation,
            translate=(translate_frac, translate_frac),
            scale=scale_range,
            shear=None,
            fill=0
        ),

        transforms.RandomCrop(crop_size),

        transforms.RandomHorizontalFlip(p=hflip_prob),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])