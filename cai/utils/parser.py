import torch


from torch import Tensor
from PIL import Image
from typing import Callable, List
import torchvision.transforms as tv_transforms
from torchvision.transforms import PILToTensor
from torchvision.transforms.functional import normalize
def _parse_label(label: str, sep: str = "|", num_classes: int = 19) -> Tensor:
    result = [int(i) for i in label.split(sep)]
    result = torch.eye(num_classes, requires_grad=False)[result].sum(0)

    return result

def _parse_images(images: List) -> Tensor:
    scale = 1.0/255.0
    pil_to_tensor = PILToTensor()
    images = torch.cat([pil_to_tensor(image) for image in images], dim=0) # concat along channel axis
    images = images * scale
    
    #images = normalize(images, [0.06438801, 0.0441467, 0.03966651, 0.06374957], [0.10712028, 0.08619478, 0.11134183, 0.10635688])
    return images

def _parse_transforms(transform_cfg: List):
    try:
        if len(transform_cfg) < 1:
            return None
        else:
            transforms = []
            for transform in transform_cfg:
                transforms.append(
                        getattr(globals()[transform["module"]],
                                transform["name"])(**transform["kwargs"]))
            return tv_transforms.Compose(transforms)
    except Exception as e:
        print(e)
        return None
