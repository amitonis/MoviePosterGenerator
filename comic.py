from diffusers import DiffusionPipeline
import torch
import gc

# Free up GPU memory before loading the model
torch.cuda.empty_cache()
gc.collect()

# Load Comic Diffusion Model with Optimized Settings
model_id = "ogkalu/Comic-Diffusion"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use FP16 for lower memory
    safety_checker=None,  # Disable safety checker for performance (Use responsibly)
).to("cuda")

# Optimize for Low VRAM
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

print(" Model loaded successfully!")
movie_description = input("🎬 Enter a 25+ word movie description: ")
import time

# Generate Poster
start_time = time.time()

poster = pipe(
    movie_description, 
    width=768,  # Higher quality resolution
    height=1024, 
    num_inference_steps=50  # High-quality generation
).images[0]

# Save & Display Poster
poster_path = "movie_poster.png"
poster.save(poster_path)
poster.show()

print(f"Movie Poster generated in {time.time() - start_time:.2f} sec")
from gpt4all import GPT4All
from PIL import Image, ImageDraw, ImageFont

# Load GPT4All Model
gpt = GPT4All("mistral-7b-openorca.Q4_0.gguf")

# Generate a Movie Title & Tagline
prompt = f"Generate a compelling movie title and tagline for this plot: {movie_description}\n\nTitle: "
response = gpt.generate(prompt)

# Extract Title & Tagline
lines = response.strip().split("\n")
movie_title = lines[0].replace("Title:", "").strip() if len(lines) > 0 else "Unknown Title"
tagline = lines[1].replace("Tagline:", "").strip() if len(lines) > 1 else "Unknown Tagline"

print(f"🎬Generated Movie Title: {movie_title}")
print(f"🎬Generated Tagline: {tagline}")

# Load the Generated Movie Poster
poster_path = "movie_poster.png"  # Path to the generated poster
poster = Image.open(poster_path)

# Convert Image to Editable Format
draw = ImageDraw.Draw(poster)
width, height = poster.size

# Load Font (Uses default if arial.ttf not found)
try:
    title_font = ImageFont.truetype("arial.ttf", 80)  # Large font for title
    tagline_font = ImageFont.truetype("arial.ttf", 40)  # Smaller font for tagline
except:
    title_font = ImageFont.load_default()
    tagline_font = ImageFont.load_default()

# Define Text Positions (Center Align)
title_x = width // 2
title_y = int(height * 0.1)  # Title at top

tagline_x = width // 2
tagline_y = int(height * 0.85)  # Tagline at bottom

# Add AI-Generated Title & Tagline
draw.text((title_x, title_y), movie_title, font=title_font, fill="white", anchor="mm")
draw.text((tagline_x, tagline_y), tagline, font=tagline_font, fill="white", anchor="mm")

# Save Final Poster with AI-Generated Text
output_path = "final_movie_poster.png"
poster.save(output_path)

print(f"Movie Poster saved as {output_path}")
poster.show()  # Display final poster