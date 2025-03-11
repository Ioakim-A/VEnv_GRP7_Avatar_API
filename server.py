from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

def crop_and_apply_image(
    generated_image_path='generated_face.png',
    mask_image_path='base_skin/base_face_mask.png',
    background_image_path='base_skin/base.jpeg',
    output_image_path='final_texture.png'
):
    # Load images
    generated_img = Image.open(generated_image_path).convert("RGBA")
    mask_img = Image.open(mask_image_path).convert("L")  # Grayscale for mask
    background_img = Image.open(background_image_path).convert("RGBA")

    # Ensure generated image matches mask dimensions
    generated_img = generated_img.resize(mask_img.size)

    # Create a new blank image with transparency
    cropped_img = Image.new("RGBA", mask_img.size)

    # Paste generated image onto cropped_img using the mask
    cropped_img.paste(generated_img, (0, 0), mask=mask_img)

    # Paste cropped image onto background
    background_img.paste(cropped_img, (0, 0), mask=cropped_img)

    # Save final composited image
    background_img.save(output_image_path)

    print(f"Final image saved to {output_image_path}")

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
    torch_dtype=torch.float16
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
    num_images = data.get("num_images", 4) 

    base_face_image = Image.open('base_skin/base_face.png').convert("RGB").resize((512, 512))

    output_image_paths = []

    generated_images = pipe(prompt_face, image=base_face_image, strength=0.8, num_images_per_prompt=num_images).images

    for idx, img in enumerate(generated_images):
        output_face_path = f"generated_face_{idx}.png"
        img.save(output_face_path)

        crop_and_apply_image(
            generated_image_path=output_face_path,
            output_image_path=f"final_output_{idx}.png"
        )

        output_image_paths.append(f"final_output_{idx}.png")

    return {"image_paths": output_image_paths}