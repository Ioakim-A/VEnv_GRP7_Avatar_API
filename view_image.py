import json
import base64
from io import BytesIO
from PIL import Image

with open("output.json", "r") as file:
    data = json.load(file)

if "images_base64" in data and len(data["images_base64"]) > 0:
    image_base64 = data["images_base64"][0] 
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image.save("output_image.png")
    print("Image saved as output_image.png")

else:
    print("No valid images found in output.json")
