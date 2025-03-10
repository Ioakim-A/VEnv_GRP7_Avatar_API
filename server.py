from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import requests

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
    prompt = data["prompt"]

    #base_prompt = 'The image is a texture which is split into multiple parts: head (top left), torso (bottom left), legs (top right), and hands (bottom right). Apply the following change to the image, leaving out unspecified parts not mentioned in this instruction:'
    prompt = f'The base image is a texture of an avatar\'s head. Modify the texture so that the character has a {prompt} hairstyle while keeping all other facial details intact.'
    #prompt = f"{base_prompt} {prompt}"
    print(prompt)

    base_image = Image.open('base_skin/base_head.jpg').convert("RGB").resize((512, 512))

    output_image = pipe(prompt, image=base_image, strength=0.7).images[0]

    output_path = "generated_skin.png"
    output_image.save(output_path)

    return {"image_path": output_path}