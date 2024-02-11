import torch
from nodes import VAEEncode


class QQ_VAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "latent": ("LATENT",), "pixels": ("IMAGE",), "vae": ("VAE",)
            }
        }

    CATEGORY = "latent"

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    def encode(self, latent=None, pixels=None, vae=None):
        if pixels is None or vae is None:
            return (latent,)
        else:
            pixels = VAEEncode.vae_encode_crop_pixels(pixels)
            t = vae.encode(pixels[:,:,:,:3])
            return ({"samples":t}, )
      
NODE_CLASS_MAPPINGS = {
	"QQ_VAEEncode": QQ_VAEEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QQ_VAEEncode": "VAEEncode_QQ"
}
