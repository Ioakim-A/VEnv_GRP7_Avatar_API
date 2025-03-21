import json
import base64
from io import BytesIO
from PIL import Image

# Load the JSON file
with open("output.json", "r") as file:
    data = json.load(file)

# Extract the Base64 image string
if "images_base64" in data and len(data["images_base64"]) > 0:
    image_base64 = data["images_base64"][0]  # Get the first image

    # Decode the Base64 string
    image_data = base64.b64decode(image_base64)

    # Open the image using PIL
    image = Image.open(BytesIO(image_data))

    # Display the image
    image.show()

    # Optionally save the image
    image.save("output_image.png")
    print("Image saved as output_image.png")

else:
    print("No valid images found in output.json")
