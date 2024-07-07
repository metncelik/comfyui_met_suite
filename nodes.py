import torch
from torchvision import transforms
from PIL import Image
from .utils import tensor2pil, pil2tensor

PARENT_CATEGORY = "MET SUITE"

class PrimitiveBBOX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "x_min": ("INT", {"default": 0, "min": 0, "step": 1}),
                    "y_min": ("INT", {"default": 0, "min": 0, "step": 1}),
                    "width": ("INT", {"default": 512, "min": 1, "step": 1}),
                    "height": ("INT", {"default": 512, "min": 1, "step": 1}),
                    }}

    CATEGORY = PARENT_CATEGORY + "/bbox"

    RETURN_TYPES = ("BBOX", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("bbox", "x_min", "y_min", "x_max", "y_max", "width", "height")
    FUNCTION = "primitive_bbox"

    def primitive_bbox(self, x_min=0, y_min=0, width=0, height=0):
        return ((x_min, y_min, width, height), x_min, y_min, x_min + width, y_min + height,width, height)

class BBOXPadding:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "bbox": ("BBOX", ),
                    "padding": ("INT", {"default": 16, "min": 0, "step": 1}),
                    "max_width": ("INT", {"default": 0, "min": 0, "step": 1}),
                    "max_height": ("INT", {"default": 0, "min": 0,  "step": 1}),
                    }}

    CATEGORY = PARENT_CATEGORY + "/bbox"

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "bbox_padding"

    def bbox_padding(self, bbox: tuple, padding=0, max_width=0, max_height=0):
        x_min, y_min, width, height = bbox

        x_min_padded = max(x_min - padding, 0)
        y_min_padded = max(y_min - padding, 0)

        new_width = width + 2 * padding
        new_height = height + 2 * padding
        
        if max_width > 0:
            new_width = min(new_width, max_width - x_min_padded)

        if max_height > 0:
            new_height = min(new_height, max_height - y_min_padded)
            
        new_bbox = (x_min_padded, y_min_padded, new_width, new_height)
        return (new_bbox, )

class BBOXResize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "bbox": ("BBOX", ),
                    "width": ("INT", {"default": 512, "min": 0, "step": 1}),
                    "height": ("INT", {"default": 512, "min": 0, "step": 1}),
                    "keep_ratio": ("BOOLEAN", {"default": True})
                    }}

    CATEGORY = PARENT_CATEGORY + "/bbox"

    RETURN_TYPES = ("BBOX", "INT", "INT")
    RETURN_NAMES = ("bbox", "new_width", "new_height")
    FUNCTION = "bbox_resize"

    def bbox_resize(self, bbox: tuple, width = 0, height = 0, keep_ratio = True):
        x_min, y_min, original_width, original_height = bbox
        
        if keep_ratio:
            if width == 0:
                width = original_width

            if height == 0:
                height = original_height

            ratio = original_width / original_height

            if width / height > ratio:
                new_height = height
                new_width = int(ratio * new_height)
            else:
                new_width = width
                new_height = int(new_width / ratio)
                
                
            x_min = x_min * new_width / original_width
            y_min = y_min * new_height / original_height
        else:
            new_width = width
            new_height = height
            
        return ((x_min, y_min, new_width, new_height), new_width, new_height)
    
class ImageResizeKeepRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "width": ("INT", {"default": 512, "min": 0,  "step": 1}),
                    "height": ("INT", {"default": 512, "min": 0,  "step": 1}),
                    }}

    CATEGORY = PARENT_CATEGORY + "/image"

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("resized_image", "new_width", "new_height" )
    FUNCTION = "resize_keep_ratio"

    def resize_keep_ratio(self, image: torch.Tensor, width=512, height=512):
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
       
# class RaiseError:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#                     "error_message": ("STRING", {"default": "ComfyUI"}),
#                     }}

#     CATEGORY = PARENT_CATEGORY

#     RETURN_TYPES = ("STRING", )
#     FUNCTION = "raise_error"
    
#     OUTPUT_NODE = True

#     def raise_error(self, error_message:str):
#         return (error_message,)
    
NODE_CLASS_MAPPINGS = {
    "PrimitiveBBOX": PrimitiveBBOX,
    "BBOXPadding": BBOXPadding,
    "BBOXResize": BBOXResize,
    "ImageResizeKeepRatio": ImageResizeKeepRatio,
    # "RaiseError": RaiseError
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimitiveBBOX": "Primitive BBOX",
    "BBOXPadding": "BBOX Padding",
    "BBOXResize": "BBOX Resize",
    "ImageResizeKeepRatio": "Image Resize Keep Ratio",
    # "RaiseError": "Raise Error"
}
