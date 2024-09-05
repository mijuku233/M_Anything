import numpy as np
import os
import imageio
import folder_paths
import zipfile
import io
import torch
from nodes import VAEEncode, InpaintModelConditioning, SaveImage, PreviewImage
from PIL import Image


class ConstrainImage_QQ:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_size": ("INT", {"default": 1920, "min": 0, "max": 16384}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constrain_image"
    CATEGORY = "QQ_Nodes"

    def constrain_image(self, images, max_size):
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")

            cur_width, cur_height = img.size
            if cur_width > cur_height:
                new_width = max_size
                new_height = int(cur_height * (max_size / cur_width))
            else:
                new_height = max_size
                new_width = int(cur_width * (max_size / cur_height))

            resized_image = img.resize((new_width, new_height), Image.LANCZOS)
            resized_image = np.array(resized_image).astype(np.float32) / 255.0
            resized_image = torch.from_numpy(resized_image)[None,]
            results.append(resized_image)
            all_images = torch.cat(results, dim=0)

        return (all_images, all_images.size(0),)


class ImageViewer_QQ:
    def __init__(self):
        self.file_name = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "file_name": ("STRING", {"default": "", "tooltip": "Fill in the file_name to save the images. Leave empty to preview the images."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "process_images"

    OUTPUT_NODE = True

    CATEGORY = "QQ_Nodes"
    DESCRIPTION = "Preview or save images."

    def process_images(self, images, file_name="", prompt=None, extra_pnginfo=None):
        if file_name:
            saver = SaveImage()
            return saver.save_images(images, file_name, prompt, extra_pnginfo)
        else:
            previewer = PreviewImage()
            return previewer.save_images(images, file_name, prompt, extra_pnginfo)


class VAEEncode_QQ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"vae": ("VAE",), "positive": ("CONDITIONING", ), "negative": ("CONDITIONING", ), },
            "optional": {"latent": ("LATENT",), "pixels": ("IMAGE",), "mask": ("MASK", ), }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "QQ_Nodes"

    def encode(self, vae, positive, negative, latent=None, pixels=None, mask=None):

        if pixels is not None:
            if mask is not None:
                positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask)
            else:
                latent = VAEEncode().encode(vae, pixels)[0]
        return (positive, negative, latent,)


class ZipImages_QQ:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "ComfyUI"})}}

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "QQ_Nodes"

    def save_images(self, images, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        zip_filename = os.path.join(full_output_folder, f'{filename_prefix}.zip')

        # Open the zip file once
        with zipfile.ZipFile(zip_filename, 'a') as zip_file:
            for idx, image in enumerate(images):
                i = 255. * image.numpy()
                img = np.clip(i, 0, 255).astype(np.uint8)
                file = f"{filename}_{idx:04}.png"

                # Ensure that the file name is unique
                while file in zip_file.namelist():
                    idx += 1
                    file = f"{filename}_{idx:04}.png"

                # Create a BytesIO object to store the image data
                with io.BytesIO() as image_bytes:
                    imageio.imwrite(image_bytes, img, format='PNG')

                    # Add the image data to the zip file
                    zip_file.writestr(os.path.join(subfolder, file), image_bytes.getvalue())

        return {"images": None}


class Pipe_QQ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "basic_pipe": ("BASIC_PIPE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("basic_pipe", "MODEL", "CLIP", "VAE", "Positive", "Negative", "Latent",)
    FUNCTION = "doit"
    CATEGORY = "QQ_Nodes"

    def doit(self, basic_pipe=(None, None, None, None, None, None),
             model=None, clip=None, vae=None, positive=None, negative=None, latent=None):

        r_model, r_clip, res_vae, r_positive, r_negative, r_latent = basic_pipe

        pipe = (model or r_model, clip or r_clip, vae or res_vae, positive or r_positive, negative or r_negative, latent or r_latent)

        return (pipe, *pipe,)


NODE_CLASS_MAPPINGS = {
    "ConstrainImage_QQ": ConstrainImage_QQ,
    "ImageViewer_QQ": ImageViewer_QQ,
    "VAEEncode_QQ": VAEEncode_QQ,
    "ZipImages_QQ": ZipImages_QQ,
    "Pipe_QQ": Pipe_QQ,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConstrainImage_QQ": "ConstrainImage_QQ",
    "ImageViewer_QQ": "ImageViewer_QQ",
    "VAEEncode_QQ": "VAEEncode_QQ",
    "ZipImages_QQ": "ZipImages_QQ",
    "Pipe_QQ": "Pipe_QQ",
}
