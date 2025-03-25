from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import base64
from io import BytesIO
from collections import Counter

import os
# Set custom cache locations if having issues with cached files taking up space in the wrong place

#os.environ["HF_HOME"] = "PATH_TO_CACHE/cache/huggingface"
#os.environ["TRANSFORMERS_CACHE"] = "PATH_TO_CACHE/cache/transformers"
#os.environ["TORCH_HOME"] = "PATH_TO_CACHE/cache/torch"
#os.environ["XDG_CACHE_HOME"] = "PATH_TO_CACHE/cache"

#print("Cache locations set:")
#print(f"HF_HOME={os.environ['HF_HOME']}")
#print(f"TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}")
#print(f"TORCH_HOME={os.environ['TORCH_HOME']}")
#print(f"XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}")

def get_dominant_color(image: Image.Image, num_colors=10) -> tuple:
        # Convert image to RGBA mode to access transparency info
        img = image.convert("RGBA")
        
        # Resize image to reduce processing time
        img = img.resize((100, 100))
        
        # Get pixel data with transparency
        pixels = list(img.getdata())
        
        # Filter out transparent pixels (alpha < 128)
        non_transparent_pixels = [(r, g, b) for r, g, b, a in pixels if a >= 128]
        
        if not non_transparent_pixels:
            # Return default color if all pixels are transparent
            return (0, 0, 0)
        
        # Count occurrences of each color
        color_count = Counter(non_transparent_pixels)
        
        # Return the most common color
        return color_count.most_common(1)[0][0]

def crop_and_apply_image(
    generated_img: Image.Image,
    mask_image_path: str = 'base_skin/base_face_mask.png',
    background_image_path: str = 'base_skin/base.jpeg'
) -> Image.Image:
    
    dominant_colour = get_dominant_color(generated_img)

    mask_img = Image.open(mask_image_path).convert("L")  # Grayscale mask
    background_img = Image.open(background_image_path).convert("RGBA")
    
    generated_img = generated_img.convert("RGBA")
    generated_img = generated_img.resize(mask_img.size)
    
    cropped_img = Image.new("RGBA", mask_img.size)
    solid_color_img = Image.new("RGBA", mask_img.size, dominant_colour)
    # Create a solid color image with the dominant color
    solid_color_img = Image.new("RGB", mask_img.size, dominant_colour)
    solid_color_img = solid_color_img.convert("RGBA")
    # Apply the mask as the alpha channel
    solid_color_img.putalpha(mask_img)
    # Use the solid color image as our base
    cropped_img = solid_color_img
    cropped_img.paste(generated_img, (0, 0), mask=mask_img)
    
    # Composite the cropped image over the background using its transparency
    background_img.paste(cropped_img, (0, 0), mask=cropped_img)
    
    return background_img


def load_pipeline():
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for inference.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon) for inference.")
    else:
        device = "cpu"
        print("No GPU detected, running on CPU (slow).")

    return StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        safety_checker=None 
    ).to(device)

pipe = load_pipeline() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/ping")
async def ping():
    return {"status": "success", "message": "Server is running"}

@app.post("/select_skin")
async def select_skin(request: Request):
    data = await request.json()
    skin_names = data["skin_names"]
    user_query = data["query"]

    skin_embeddings = model.encode(skin_names, convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, skin_embeddings)

    best_match_idx = torch.argmax(similarities).item()
    chosen_skin = skin_names[best_match_idx]

    return {"chosen_skin": chosen_skin}

@app.post("/generate_skin_image_face")
async def generate_skin_image_face(request: Request):
    data = await request.json()
    prompt_face = data["prompt_face"]
    num_images = data.get("num_images", 4)

    prompt_face = prompt_face.replace(".", "").replace(",", "")
    if 'face' not in prompt_face:
        prompt_face = f'{prompt_face} face'

    base_face_image = Image.open('base_skin/base_face_closeup.png').convert("RGB").resize((512, 512))

    generated_images = pipe(prompt_face,
                            image=base_face_image,
                            strength=0.8,
                            num_inference_steps=100,
                            num_images_per_prompt=num_images).images

    output_images_base64 = []

    for img in generated_images:
        final_img = crop_and_apply_image(img)

        buffer = BytesIO()
        final_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        output_images_base64.append(img_str)

    return JSONResponse(content={"images_base64": output_images_base64}) 

@app.post("/generate_skin_image_torso")
async def generate_skin_image_torso(request: Request):
    data = await request.json()
    prompt_torso = data["prompt_torso"]
    num_images = data.get("num_images", 4)

    prompt_torso = prompt_torso.replace(".", "").replace(",", "").replace("-", "")
    if 'top' not in prompt_torso:
        prompt_torso = f'{prompt_torso} top'

    base_torso_image = Image.open('base_skin/torso/base_front_torso_white_bg.png').convert("RGB").resize((512, 512))

    generated_images = pipe(prompt_torso,
                           image=base_torso_image,
                           strength=0.8,
                           num_inference_steps=100,
                           num_images_per_prompt=num_images).images

    output_images_base64 = []

    for img in generated_images:
        front_img = crop_and_apply_image(
            img, 
            mask_image_path='base_skin/torso/base_front_torso_mask.png',
            background_image_path='base_skin/torso/base_front_torso_mask.png'
        )

        # Apply dominant color to the mask and save it
        dominant_color = get_dominant_color(front_img)
        mask_path = 'base_skin/torso/base_torso_mask.png'
        colored_mask_path = 'base_skin/torso/base_torso_mask_coloured.png'
        
        mask_img = Image.open(mask_path).convert("RGBA")
        
        solid_color = Image.new("RGB", mask_img.size, dominant_color)
        solid_color = solid_color.convert("RGBA")

        r, g, b, a = mask_img.split()
        solid_color.putalpha(a)
        
        solid_color.save(colored_mask_path)
        
        torso_img = crop_and_apply_image(
            front_img, 
            mask_image_path='base_skin/torso/base_front_torso_mask.png',
            background_image_path='base_skin/torso/base_torso_mask_coloured.png'
        )

        final_img = crop_and_apply_image(
            torso_img, 
            mask_image_path='base_skin/torso/base_torso_mask.png',
            background_image_path='base_skin/torso/base.png'
        )

        buffer = BytesIO()
        final_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        output_images_base64.append(img_str)

    return JSONResponse(content={"images_base64": output_images_base64})