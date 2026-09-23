import os
import certifi
import logging
import traceback

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import QwenImageEditPipeline
import torch
from io import BytesIO
from PIL import Image
import base64

app = FastAPI(title="Local Qwen-Image-Edit API")
logger = logging.getLogger(__name__)

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    low_cpu_mem_usage=True,
    offload_folder="offload",
)

# PBR map generation prompts keyed by map type.
# Each takes the albedo texture as input image and converts it.
_PBR_PROMPTS = {
    "normal": (
        "Convert this texture into a tangent-space normal map. "
        "Output a blue-purple image where colour encodes surface normals: "
        "flat/smooth areas are pure #8080FF (128,128,255 RGB), "
        "raised bumps shift toward yellow-green, recessed areas toward dark-blue. "
        "No albedo colour, no shading — pure normal-map encoding only."
    ),
    "roughness": (
        "Convert this texture into a grayscale roughness map for PBR rendering. "
        "White (255) = fully rough/matte surface, black (0) = perfectly smooth/glossy. "
        "Derive roughness from the material's visual appearance. "
        "Single-channel grayscale, no colour tint, no shading."
    ),
    "metallic": (
        "Convert this texture into a grayscale metallic map for PBR rendering. "
        "White (255) = fully metallic material, black (0) = non-metallic/dielectric. "
        "Most painted plaster, wood, and fabric surfaces should be near black. "
        "Single-channel grayscale, no colour tint, no shading."
    ),
}

_PBR_NEG = "colour, shading, shadows, lighting, albedo, photorealistic scene, blurry"


class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = " "
    num_images: int = 1
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    reference_image: str | None = None   # base64-encoded; if absent a white canvas is used


class PBRRequest(BaseModel):
    albedo_image: str          # base64-encoded albedo texture (PNG/JPEG)
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20   # 3 passes; keep low to avoid client timeout
    true_cfg_scale: float = 4.0


def _decode_image(b64: str, width: int, height: int) -> Image.Image:
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img.resize((width, height))


def _white_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (255, 255, 255))


def _to_b64(img: Image.Image, width: int, height: int) -> str:
    out = img.resize((width, height))
    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/generate")
async def generate(req: PromptRequest):
    try:
        ref_img = (
            _decode_image(req.reference_image, req.width, req.height)
            if req.reference_image
            else _white_canvas(req.width, req.height)
        )

        results = []
        with torch.inference_mode():
            for _ in range(req.num_images):
                output = pipe(
                    image=ref_img,
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    num_inference_steps=req.num_inference_steps,
                    true_cfg_scale=req.true_cfg_scale,
                )
                results.append(_to_b64(output.images[0], req.width, req.height))

        return {"images": results}
    except Exception as e:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}") from e


@app.post("/generate_pbr")
async def generate_pbr(req: PBRRequest):
    """
    Given a base64-encoded albedo texture, generate aligned PBR maps:
      normal    — tangent-space normal map (RGB, blue-purple base)
      roughness — grayscale roughness map  (white=rough, black=smooth)
      metallic  — grayscale metallic map   (white=metal, black=dielectric)

    Returns {"normal": b64, "roughness": b64, "metallic": b64}.
    """
    try:
        albedo = _decode_image(req.albedo_image, req.width, req.height)
        maps: dict[str, str] = {}

        with torch.inference_mode():
            for map_type, prompt in _PBR_PROMPTS.items():
                output = pipe(
                    image=albedo,
                    prompt=prompt,
                    negative_prompt=_PBR_NEG,
                    num_inference_steps=req.num_inference_steps,
                    true_cfg_scale=req.true_cfg_scale,
                )
                maps[map_type] = _to_b64(output.images[0], req.width, req.height)

        return maps
    except Exception as e:
        logger.exception("PBR generation failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}") from e
