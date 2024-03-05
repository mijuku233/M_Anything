import numpy as np
import os
import imageio
import folder_paths
import zipfile
import io
import torch
from nodes import VAEEncode

class QQ_VAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "latent": ("LATENT",), "pixels": ("IMAGE",), "vae": ("VAE",)
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "QQ_Nodes"

    def encode(self, latent=None, pixels=None, vae=None):
        if pixels is None or vae is None:
            return (latent,)
        else:
            t = vae.encode(pixels[:, :, :, :3])
            return ({"samples": t},)


class ZipImages:
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

class QQ_Pipe:
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
            },
        }

    RETURN_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("basic_pipe", "MODEL", "CLIP", "VAE", "Positive", "Negative",)
    FUNCTION = "doit"
    CATEGORY = "QQ_Nodes"

    def doit(self, basic_pipe=(None, None, None, None, None),
             model=None, clip=None, vae=None, positive=None, negative=None):

        r_model, r_clip, res_vae, r_positive, r_negative = basic_pipe

        pipe = (model or r_model, clip or r_clip, vae or res_vae, positive or r_positive, negative or r_negative)

        return (pipe, *pipe,)


NODE_CLASS_MAPPINGS = {
    "QQ_VAEEncode": QQ_VAEEncode,
    "ZipImages": ZipImages,
    "QQ_Pipe": QQ_Pipe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QQ_VAEEncode": "VAEEncode_QQ",
    "ZipImages": "ZipImages",
    "QQ_Pipe": "QQ_Pipe",
}
