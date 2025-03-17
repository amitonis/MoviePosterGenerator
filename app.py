from flask import Flask, render_template, request, send_from_directory
import torch
from diffusers import StableDiffusionPipeline
import os

app = Flask(__name__)
output_path = "static/generated_image.png"

# Load Stable Diffusion model (only once at startup)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

def generate_image(prompt):
    image = pipe(prompt).images[0]  # Generate image from prompt
    image.save(output_path)  # Save the generated image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        generate_image(prompt)
        return render_template("index.html", image_url=output_path)
    return render_template("index.html", image_url=None)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
