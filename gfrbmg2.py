import torch, os
import torch.nn.functional as F
import folder_paths
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from torchvision.transforms.functional import normalize
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Добавляем путь к моделям ComfyUI
folder_paths.add_model_folder_path("rmbg_models", os.path.join(folder_paths.models_dir, "RMBG-2.0"))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

class GFrbmg2:
    def __init__(self):
        self.model = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image_rgba", "mask", "image_black")
    FUNCTION = "remove_background"
    CATEGORY = "🐵 GorillaFrame/Image"
  
    def remove_background(self, image, invert_mask):
        if self.model is None:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                os.path.join(folder_paths.models_dir, "RMBG-2.0"),
                trust_remote_code=True,
                local_files_only=True
            )
            self.model.to(device)
            self.model.eval()

        processed_images = []
        processed_masks = []
        processed_blacks = []

        for img in image:
            orig_image = tensor2pil(img)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            if torch.cuda.is_available():
                im_tensor=im_tensor.cuda()

            with torch.no_grad():
                result = self.model(im_tensor)[-1].sigmoid().cpu()
            
            result = result[0].squeeze()
            result = F.interpolate(result.unsqueeze(0).unsqueeze(0), size=(h,w), mode='bilinear').squeeze()
            
            if invert_mask:
                result = 1 - result
                
            mask_pil = tensor2pil(result)
            
            # RGBA image
            rgba_image = orig_image.copy()
            rgba_image.putalpha(mask_pil)
            
            # Black background image
            black_image = Image.new('RGB', orig_image.size, (0, 0, 0))
            black_image.paste(orig_image, mask=mask_pil)

            processed_images.append(pil2tensor(rgba_image))
            processed_masks.append(pil2tensor(mask_pil))
            processed_blacks.append(pil2tensor(black_image))

        new_images = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)
        new_blacks = torch.cat(processed_blacks, dim=0)

        return new_images, new_masks, new_blacks