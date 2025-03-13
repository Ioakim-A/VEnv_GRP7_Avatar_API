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

def get_dominant_color(image: Image.Image, num_colors=10) -> tuple:
        """
        Returns the most dominant RGB color in the image.
        
        Args:
            image: PIL Image object
            num_colors: Number of colors to quantize to before finding dominant
        
        Returns:
            tuple: (R, G, B) tuple representing the most frequent color
        """
        # Convert image to RGB mode if it's not already
        img = image.convert("RGB")
        
        # Resize image to reduce processing time
        img = img.resize((100, 100))
        
        # Quantize the image to a limited set of colors
        quantized = img.quantize(colors=num_colors)
        quantized = quantized.convert("RGB")
        
        # Get color data and count occurrences
        pixels = list(quantized.getdata())
        
        color_count = Counter(pixels)
        
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
    
    # Create a new blank image for the cropped output
    cropped_img = Image.new("RGBA", mask_img.size)
    # Create a solid color background with the dominant color
    solid_color_img = Image.new("RGBA", mask_img.size, dominant_colour)
    # Create a solid color image with the dominant color
    solid_color_img = Image.new("RGB", mask_img.size, dominant_colour)
    # Convert to RGBA and use the mask as the alpha channel
    solid_color_img = solid_color_img.convert("RGBA")
    # Apply the mask as the alpha channel
    solid_color_img.putalpha(mask_img)
    # Use the solid color image as our base
    cropped_img = solid_color_img
    # Paste generated image onto cropped image using the mask
    cropped_img.paste(generated_img, (0, 0), mask=mask_img)
    
    # Composite the cropped image over the background using its transparency
    background_img.paste(cropped_img, (0, 0), mask=cropped_img)
    
    return background_img

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("mps")

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

@app.post("/generate_skin_image")
async def generate_skin_image(request: Request):
    data = await request.json()
    prompt_face = data["prompt_face"]
    num_images = data.get("num_images", 3)

    base_face_image = Image.open('base_skin/base_face_closeup.png').convert("RGB").resize((512, 512))

    generated_images = pipe(prompt_face,
                            image=base_face_image,
                            strength=0.8,
                            num_inference_steps=40,
                            num_images_per_prompt=num_images).images

    output_images_base64 = []

    for img in generated_images:
        final_img = crop_and_apply_image(img)

        buffer = BytesIO()
        final_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        output_images_base64.append(img_str)

    return JSONResponse(content={"images_base64": output_images_base64})