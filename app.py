from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import os
import threading

app = Flask(__name__)

output_path = "static/generated_image.png"

# Load AI models
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
text_generator = pipeline("text-generation", model="gpt2")

progress = {"status": "idle"}  # Track progress

def generate_movie_poster(prompt):
    """Generate a movie poster."""
    global progress
    progress["status"] = "Generating poster..."
    image = pipe(prompt).images[0]
    image.save(output_path)
    progress["status"] = "done"

def generate_movie_title(prompt):
    """Generate a movie title based on the input description."""
    global progress
    progress["status"] = "Generating title..."
    
    response = text_generator(prompt, max_new_tokens=15, num_return_sequences=1)
    progress["status"] = "done"
    
    return response[0]["generated_text"].strip()


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", image_url=None, movie_title=None)

@app.route("/generate", methods=["POST"])
def generate():
    global progress
    data = request.json
    prompt = data.get("prompt")

    # Run both tasks in threads for a better user experience
    title_thread = threading.Thread(target=lambda: progress.update({"title": generate_movie_title(prompt)}))
    poster_thread = threading.Thread(target=lambda: generate_movie_poster(prompt))

    title_thread.start()
    poster_thread.start()
    title_thread.join()
    poster_thread.join()

    return jsonify({"image_url": output_path, "movie_title": progress["title"]})

@app.route("/progress", methods=["GET"])
def get_progress():
    return jsonify(progress)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
