import torch
from torchvision import transforms
from PIL import Image
from .utils import tensor2pil, pil2tensor

DEFAULT_CATEGORY = "MET SUITE"

class BBOXPadding:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "bbox": ("BBOX", ),
                    "padding": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                    }}

    CATEGORY = DEFAULT_CATEGORY

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "bbox_padding"

    def bbox_padding(self, bbox: tuple, padding=0, max_width=0, max_height=0):
        x_min, y_min, x_max, y_max = bbox

        x_min_padded = max(x_min - padding, 0)
        y_min_padded = max(y_min - padding, 0)
        x_max_padded = x_max + padding
        y_max_padded = y_max + padding

        if max_width > 0:
            x_max_padded = min(x_max_padded, max_width)
            x_min_padded = min(x_min_padded, x_max_padded)

        if max_height > 0:
            y_max_padded = min(y_max_padded, max_height)
            y_min_padded = min(y_min_padded, y_max_padded)

        new_bbox = (x_min_padded, y_min_padded, x_max_padded, y_max_padded)
        return new_bbox
        
        
class ResizeKeepRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "width": ("INT", {"default": 512, "min": 0,  "step": 1}),
                    "height": ("INT", {"default": 512, "min": 0,  "step": 1}),
                    }}

    CATEGORY = DEFAULT_CATEGORY

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("resized_image", "new_width", "new_height" )
    FUNCTION = "resize_keep_ratio"

    def resize_keep_ratio(self, image: torch.Tensor, width=512, height=768, fill_color=(0, 0, 0)):
        image = tensor2pil(image)[0]
        original_width, original_height = image.size
        ratio = original_width / original_height

        if width / height > ratio:
            new_height = height
            new_width = int(ratio * new_height)
        else:
            new_width = width
            new_height = int(new_width / ratio)

        resized_im = image.resize((new_width, new_height))

        resized_tensor = pil2tensor(resized_im)

        return (resized_tensor, new_width, new_height)   

class BBOXResize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "bbox": ("BBOX", ),
                    "width": ("INT", {"default": 512, "min": 0, "max": 255, "step": 1}),
                    "height": ("INT", {"default": 512, "min": 0, "max": 255, "step": 1}),
                    "keep_ratio": ("BOOLEAN", {"default": True})
                    }}

    CATEGORY = DEFAULT_CATEGORY

    RETURN_TYPES = ("BBOX", "INT", "INT")
    RETURN_NAMES = ("bbox", "new_width", "new_height")
    FUNCTION = "bbox_resize"

    def bbox_resize(self, bbox: tuple, width = 0, height = 0, keep_ratio = True):
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly four elements.")
        
        x_min, y_min, x_max, y_max = bbox
        
        if keep_ratio:
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            ratio = bbox_width / bbox_height
            
            if width / height > ratio:
                new_height = height
                new_width = int(ratio * new_height)
            else:
                new_width = width
                new_height = int(new_width / ratio)

            new_x_min = x_min
            new_y_min = y_min
            new_x_max = x_min + new_width
            new_y_max = y_min + new_height
        else:
            new_x_min = x_min
            new_y_min = y_min
            new_x_max = x_min + width
            new_y_max = y_min + height
        
        new_bbox = (new_x_min, new_y_min, new_x_max, new_y_max)
        return (new_bbox,)
       

    
NODE_CLASS_MAPPINGS = {
    "BBOXPadding": BBOXPadding,
    "BBOXResize": BBOXResize,
    "ResizeKeepRatio": ResizeKeepRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BBOXPadding": "BBOX Padding",
    "BBOXResize": "BBOX Resize",
    "ResizeKeepRatio": "Resize Image Keep Ratio"
}
