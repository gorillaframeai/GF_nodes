import torch
import torch.nn.functional as F
import folder_paths
import os
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
import random
from huggingface_hub import hf_hub_download, HfApi

device = "cuda" if torch.cuda.is_available() else "cpu"

# Добавляем путь к моделям ComfyUI
folder_paths.add_model_folder_path("rmbg_models", os.path.join(folder_paths.models_dir, "RMBG", "RMBG-2.0"))
folder_paths.add_model_folder_path("ben2_models", os.path.join(folder_paths.models_dir, "RMBG", "BEN2"))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image, interpolation_method):
    image = image.convert('RGB')
    w, h = image.size
    new_w = (w + 63) // 64 * 64
    new_h = (h + 63) // 64 * 64
    image = image.resize((new_w, new_h), interpolation_method)
    return image

def center_crop_or_pad(image, target_size, scale=1.0, rotation=0.0):
    # Преобразуем scale: отрицательные значения уменьшают, положительные увеличивают
    actual_scale = 1.0 + (scale / 10.0)  # От 0.0 до 2.0

    # Получаем размеры
    target_w, target_h = target_size
    w, h = image.size

    # Вычисляем масштаб для вписывания по ширине
    width_ratio = target_w / w
    new_w = target_w
    new_h = int(h * width_ratio)

    # Применяем дополнительное масштабирование от пользователя
    new_w = int(new_w * actual_scale)
    new_h = int(new_h * actual_scale)

    # Масштабируем изображение
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Поворачиваем изображение на заданный угол
    if rotation != 0:
        image = image.rotate(-rotation, expand=True)

    # Получаем новые размеры после поворота
    w, h = image.size

    # Вычисляем отступы для центрирования
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # Обрезаем или дополняем изображение
    if left < 0 or top < 0 or right > w or bottom > h:
        # Создаем новое изображение нужного размера
        new_image = Image.new(image.mode, target_size, (0, 0, 0, 0))
        # Вычисляем координаты для вставки
        paste_left = max(0, -left)
        paste_top = max(0, -top)
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(w, right)
        crop_bottom = min(h, bottom)
        # Вставляем обрезанную часть изображения
        new_image.paste(image.crop((crop_left, crop_top, crop_right, crop_bottom)),
                       (paste_left, paste_top))
        return new_image
    else:
        return image.crop((left, top, right, bottom))

class GFrbmg2Plus:
    _model_instance = None

    @classmethod
    def get_model(cls):
        if cls._model_instance is None:
            cls.initialize_model()
        return cls._model_instance

    @classmethod
    def initialize_model(cls):
        model_path = os.path.join(folder_paths.models_dir, "RMBG", "RMBG-2.0")
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        # Получаем список всех файлов в репозитории
        api = HfApi()
        files = api.list_repo_files("briaai/RMBG-2.0")
        # Скачиваем только файлы из корневой директории
        for file in files:
            # Проверяем что файл в корне (нет слешей в пути)
            if '/' not in file:
                if file.endswith(('.json', '.py', '.safetensors')):
                    print(f"Downloading {file}...")
                    hf_hub_download(
                        repo_id="briaai/RMBG-2.0",
                        filename=file,
                        local_dir=model_path
                    )
        cls._model_instance = AutoModelForImageSegmentation.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        cls._model_instance.to(device)
        cls._model_instance.eval()
        print("Model loaded successfully.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "expand_mask": ("FLOAT", {"default": 0.0, "min": -255, "max": 255, "step": 0.1}),
                "blur_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255, "step": 0.1}),
                "sticker_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255, "step": 0.1}),
                "sticker_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255, "step": 0.1}),
                "sticker_color": ("COLOR", {"default": "#000000"}),
                "background_color": ("COLOR", {"default": "#000000"}),
                "use_original_bg": ("BOOLEAN", {"default": False}),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "slider"
                }),
                "interpolation_method": (["Lanczos", "Bicubic", "Bilinear", "Nearest"], {"default": "Lanczos"})
            },
            "optional": {
                "bg_image": ("IMAGE",),
                "bg_image_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("image_rgba", "mask", "image", "sticker_mask")
    FUNCTION = "remove_background"
    CATEGORY = "🐵 GorillaFrame/Image"

    def __init__(self):
        self.model = self.get_model()

    def clean_mask(self, mask, expand_mask, blur_weight):
        try:
            if expand_mask == 0 and blur_weight == 0:
                return mask

            mask_np = np.array(mask)
            mask_tensor = torch.from_numpy(mask_np).float().cuda()

            if mask_tensor is None or mask_tensor.numel() == 0:
                return mask

            if blur_weight > 0:
                print("Applying blur to mask...")
                kernel_size = max(3, int(blur_weight * 20) | 1)
                sigma = blur_weight * 5
                padding = kernel_size // 2
                mask_blur = mask_tensor.unsqueeze(0).unsqueeze(0)

                coords = torch.arange(kernel_size, device='cuda').float() - padding
                x, y = torch.meshgrid(coords, coords)
                kernel_2d = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))

                kernel_2d = kernel_2d / kernel_2d.sum()
                kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

                mask_blur = F.conv2d(mask_blur, kernel_2d, padding=padding)
                mask_tensor = mask_blur.squeeze()

            if expand_mask != 0:
                print("Expanding mask...")
                kernel_size = max(3, int(abs(expand_mask) * 10) | 1)
                padding = kernel_size // 2

                if expand_mask > 0:
                    pool = torch.nn.MaxPool2d(kernel_size, stride=1, padding=padding)
                    mask_tensor = pool(mask_tensor.unsqueeze(0)).squeeze(0)
                else:
                    pool = torch.nn.MaxPool2d(kernel_size, stride=1, padding=padding)
                    mask_tensor = -pool(-mask_tensor.unsqueeze(0)).squeeze(0)

            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
            # Очищаем CUDA память
            del mask_tensor
            torch.cuda.empty_cache()

            return Image.fromarray(mask_np)

        except Exception as e:
            print(f"Error in clean_mask: {e}")
            return mask

    def remove_background(self, image, invert_mask, expand_mask, blur_weight, sticker_size, sticker_blur, sticker_color, background_color, bg_image=None, bg_image_scale=1.0, use_original_bg=False, rotation=0, interpolation_method="Lanczos"):
        print("Starting background removal process...")
        # Конвертируем цвета из HEX в RGB
        sticker_rgb = tuple(int(sticker_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        background_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        processed_images = []
        processed_masks = []
        processed_blacks = []
        processed_sticker_masks = []

        center_pil = None
        if bg_image is not None and len(bg_image) > 0:
            center_pil = tensor2pil(bg_image[0])

        for img in image:
            orig_image = tensor2pil(img)
            w, h = orig_image.size

            # Проверка минимального размера
            if w < 64 or h < 64:
                print(f"Warning: Image size {w}x{h} is too small, might cause issues")

            # Проверка максимального размера для предотвращения проблем с памятью
            max_size = 4096
            if w > max_size or h > max_size:
                print(f"Warning: Image size {w}x{h} is very large, might cause memory issues")

            # Создаем базовые изображения
            rgba_image = Image.new('RGBA', (w, h), (0, 0, 0, 0))

            # Обрабатываем фон
            if use_original_bg:
                black_image = orig_image.copy()
            else:
                if bg_image is not None and len(bg_image) > 0:
                    bg_pil = tensor2pil(bg_image[0])
                    black_image = center_crop_or_pad(bg_pil, (w, h), bg_image_scale, rotation)
                else:
                    black_image = Image.new('RGB', (w, h), background_color)

            # Накладываем RGBA на фон
            black_image.paste(rgba_image, (0, 0), rgba_image)

            # Получаем маску
            interpolation_mapping = {
                "Lanczos": Image.LANCZOS,
                "Bicubic": Image.BICUBIC,
                "Bilinear": Image.BILINEAR,
                "Nearest": Image.NEAREST
            }
            interpolation = interpolation_mapping.get(interpolation_method, Image.LANCZOS)
            image_resized = resize_image(orig_image, interpolation)
            im_np = np.array(image_resized)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            if torch.cuda.is_available():
                im_tensor=im_tensor.cuda()

            with torch.no_grad():
                print("Running model inference...")
                result = self.model(im_tensor)[-1].sigmoid().cpu()

            result = result[0].squeeze()
            result = F.interpolate(result.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear').squeeze()

            mask_pil = tensor2pil(result)
            mask_pil = self.clean_mask(mask_pil, expand_mask, blur_weight)

            if invert_mask:
                print("Inverting mask...")
                mask_np = np.array(mask_pil)
                mask_np = 255 - mask_np
                mask_pil = Image.fromarray(mask_np)

            if orig_image.mode != 'RGBA':
                orig_image = orig_image.convert('RGBA')

            # Накладываем изображение с маской
            rgba_image.paste(orig_image, (0, 0), mask_pil)
            black_image.paste(orig_image, (0, 0), mask_pil)

            # Обрабатываем стикер
            if sticker_size > 0:
                print("Processing sticker...")
                mask_np = np.array(mask_pil)

                # Убедимся, что маска бинарная для точного определения границы
                _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

                # Находим границу маски (это даст нам точную линию по краю)
                edges = cv2.Canny(binary_mask, 100, 200)

                # Находим контуры по границе
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # Создаем пустую маску
                main_contour_mask = np.zeros_like(mask_np)

                # Рисуем контур с равномерной толщиной точно по границе
                cv2.drawContours(main_contour_mask, contours, -1, 255, int(sticker_size), lineType=cv2.LINE_AA)

                # Применяем размытие если нужно
                if sticker_blur > 0:
                    blur_size = int(sticker_blur * 2) * 2 + 1
                    main_contour_mask = cv2.GaussianBlur(main_contour_mask, (blur_size, blur_size), 0)

                sticker_mask = Image.fromarray(main_contour_mask)

                # Применяем стикер
                sticker = Image.new('RGBA', orig_image.size, (*sticker_rgb, 255))
                rgba_image.paste(sticker, mask=sticker_mask)
                black_sticker = Image.new('RGB', orig_image.size, sticker_rgb)
                black_image.paste(black_sticker, mask=sticker_mask)

                processed_sticker_masks.append(pil2tensor(sticker_mask))
            else:
                # Если sticker_size = 0, тоже создаем пустую маску
                empty_mask = Image.new('L', orig_image.size, 0)
                processed_sticker_masks.append(pil2tensor(empty_mask))

            processed_images.append(pil2tensor(rgba_image))
            processed_masks.append(pil2tensor(mask_pil))
            processed_blacks.append(pil2tensor(black_image))

        print("Background removal process completed.")
        return (torch.cat(processed_images, dim=0),
                torch.cat(processed_masks, dim=0),
                torch.cat(processed_blacks, dim=0),
                torch.cat(processed_sticker_masks, dim=0))

NODE_CLASS_MAPPINGS = {
    "GFrbmg2Plus": GFrbmg2Plus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GFrbmg2Plus": "🐵 GF Remove Background (GFrbmg2Plus)"
}