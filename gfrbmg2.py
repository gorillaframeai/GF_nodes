import sys
import os
import torch
import folder_paths
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
import subprocess
import torch.nn.functional as F

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Добавляем путь к моделям ComfyUI!
folder_paths.add_model_folder_path("rmbg_models", os.path.join(folder_paths.models_dir, "RMBG", "RMBG-2.0"))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

def clone_model_if_not_exists(model_path, repo_url):
    if not os.path.exists(model_path):
        subprocess.run(["git", "clone", repo_url, model_path])

class GFrbmg2:
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "chroma_key_color": (["Black", "White", "Green", "Red", "Blue", "Gray"], {"default": "Black"}),
                "postprocess_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "edge_enhancement": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1
                }),
                "blur_edges": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5
                }),
                "expand_mask": ("FLOAT", {
                    "default": 0.0,
                    "min": -50.0,  # Сжатие маски
                    "max": 50.0,   # Расширение маски
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image_rgba", "image", "mask_rgb", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "🐵 GorillaFrame/Image"

    def clean_mask(self, mask, strength, edge_enhancement, blur_edges, expand_mask):
        try:
            if strength == 0 and edge_enhancement == 0 and blur_edges == 0 and expand_mask == 0:
                return mask

            mask_np = np.array(mask)
            if mask_np is None or mask_np.size == 0:
                return mask

            if expand_mask != 0:
                kernel_size = int(5 * abs(expand_mask))
                kernel_size = max(1, kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = np.ones((kernel_size, kernel_size), np.uint8)

                if expand_mask > 0:
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                else:
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)

            if edge_enhancement > 0:
                lower_threshold = int(100 * edge_enhancement)
                upper_threshold = int(200 * edge_enhancement)
                edges = cv2.Canny(mask_np, lower_threshold, upper_threshold)

                if blur_edges > 0:
                    blur_size = int(5 * blur_edges)
                    if blur_size % 2 == 0:
                        blur_size += 1
                    edges = cv2.GaussianBlur(edges, (blur_size, blur_size), 0)

                mask_np = cv2.bitwise_or(mask_np, edges)

            if strength > 0:
                kernel_size = int(5 * strength)
                kernel_size = max(1, kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                iterations = int(strength)

                original_mask = mask_np.copy()
                mask_smoothed = cv2.medianBlur(mask_np, kernel_size)
                mask_dilated = cv2.dilate(mask_smoothed, kernel, iterations=iterations)
                mask_eroded = cv2.erode(mask_dilated, kernel, iterations=iterations)
                mask_np = cv2.bitwise_and(mask_eroded, original_mask)

            cleaned_mask = Image.fromarray(mask_np)
            return cleaned_mask

        except:
            return mask

    def initialize_model(self):
        model_path = os.path.join(folder_paths.models_dir, "RMBG", "RMBG-2.0")
        clone_model_if_not_exists(model_path, "https://huggingface.co/briaai/RMBG-2.0")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.to(device)
        self.model.eval()

    def remove_background(self, image, invert_mask, chroma_key_color, postprocess_strength, edge_enhancement, blur_edges, expand_mask):
        self.initialize_model()

        processed_images = []
        processed_blacks = []
        processed_masks = []
        processed_masks_rgb = []

        for img in image:
            orig_image = tensor2pil(img)
            w, h = orig_image.size

            image = resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = torch.unsqueeze(im_tensor, 0)
            im_tensor = torch.divide(im_tensor, 255.0)
            im_tensor = normalize(im_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if torch.cuda.is_available():
                im_tensor = im_tensor.cuda()

            with torch.no_grad():
                result = self.model(im_tensor)[-1].sigmoid().cpu()

            result = result[0].squeeze()
            result = F.interpolate(result.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True).squeeze()

            mask_pil = tensor2pil(result)
            mask_pil = self.clean_mask(mask_pil, postprocess_strength, edge_enhancement, blur_edges, expand_mask)

            if invert_mask:
                mask_np = np.array(mask_pil)
                mask_np = 255 - mask_np
                mask_pil = Image.fromarray(mask_np)

            rgba_image = orig_image.copy()
            rgba_image.putalpha(mask_pil)

            background_color_value = {
                "Black": (0, 0, 0),
                "White": (255, 255, 255),
                "Green": (0, 255, 0),
                "Red": (255, 0, 0),
                "Blue": (0, 0, 255),
                "Gray": (128, 128, 128)
            }[chroma_key_color]
            background_image = Image.new('RGB', orig_image.size, background_color_value)
            background_image.paste(orig_image, mask=mask_pil)

            mask_rgb = Image.new('RGB', orig_image.size, (0, 0, 0))
            mask_rgb.paste(Image.fromarray(np.array(mask_pil).astype(np.uint8)), mask=mask_pil)

            processed_images.append(pil2tensor(rgba_image))
            processed_blacks.append(pil2tensor(background_image))
            processed_masks.append(pil2tensor(mask_pil))
            processed_masks_rgb.append(pil2tensor(mask_rgb))

        new_images = torch.cat(processed_images, dim=0)
        new_blacks = torch.cat(processed_blacks, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)
        new_masks_rgb = torch.cat(processed_masks_rgb, dim=0)

        return new_images, new_blacks, new_masks_rgb, new_masks