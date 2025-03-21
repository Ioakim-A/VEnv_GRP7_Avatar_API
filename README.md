# COMP0113 Group 7 Local Server for Sentence Transformer and Stable Diffusion
This repository contains the code and resources required to setup and run the models needed for our Avatar Unity Project.

## Setup
* **Python 3.12.9** was used.

* Apple's ML-Stable-Diffusion was used but feel free to use your own model with image&prompt-to-image generation. To install:
    1. `git clone https://github.com/apple/ml-stable-diffusion.git`
    2. `cd ml-stable-diffusion`
    3. `pip install -e .`

* Then switch back to this repository's directory.

* Setup a virtual environment and activate it
* Run `pip install -r requirements.txt`

* Start the server by running: `uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2` Note that if this is done for the first time, the models will need to be downloaded which may take a while.

## Usage
* Since this runs the server locally, requests need to be made using devices on the same network. You will need to find the IP address of your device. On Apple devices this can be done with: `ipconfig getifaddr en0`

* Use this address to make API requests: `http://<your IP>:8000/select_skin`

# Lab machine setup

* To setup the lab machine:
    1. `pip uninstall torch`
    1. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
    2. `pip install -r requirements_cuda.txt`
* To get the IP address: `ip a | grep 'inet'`. Make sure it is the ipv4 address not `127.0.0.1`
* Setup ngrok 

Run ngrok
* ngrok http 8000

# Example curl request
curl -X POST "https://c037-128-16-11-100.ngrok-free.app/generate_skin_image_face" \
  -H "Content-Type: application/json" \
  -d '{"prompt_face": "futuristic robot face, metallic, glowing eyes", "num_images": 1}' \
  -o output.json
