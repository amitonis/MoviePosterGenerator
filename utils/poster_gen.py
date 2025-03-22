# utils/poster_gen.py
import torch # type: ignore
from diffusers import StableDiffusionPipeline # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image, ImageDraw, ImageFont # type: ignore
import os
import uuid

# Load the models once
pipe = StableDiffusionPipeline.from_pretrained("C:/Users/amitm/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b", torch_dtype=torch.float16)
pipe.to("cuda")
pipe.enable_attention_slicing()

caption_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

FONT_PATH_TITLE = "static/fonts/impact.ttf"
FONT_PATH_TAGLINE = "static/fonts/arial.ttf"

OUTPUT_DIR = "generated/posters"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_title_and_tagline(description):
    prompt = f"Generate a movie title and tagline based on this description: {description}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = caption_model.generate(**inputs, max_new_tokens=40)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if ":" in response:
        title, tagline = response.split(":", 1)
    elif "-" in response:
        title, tagline = response.split("-", 1)
    else:
        title = response
        tagline = ""
    return title.strip(), tagline.strip()

def generate_image(description):
    image = pipe(description).images[0]
    return image

def overlay_text(image, title, tagline):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    title_font = ImageFont.truetype(FONT_PATH_TITLE, size=60)
    tagline_font = ImageFont.truetype(FONT_PATH_TAGLINE, size=32)

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    tagline_bbox = draw.textbbox((0, 0), tagline, font=tagline_font)

    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]

    tagline_w = tagline_bbox[2] - tagline_bbox[0]
    tagline_h = tagline_bbox[3] - tagline_bbox[1]


    draw.text(((width - title_w) / 2, 40), title, font=title_font, fill="white")
    draw.text(((width - tagline_w) / 2, height - tagline_h - 40), tagline, font=tagline_font, fill="white")

    return image

def generate_poster(description):
    title, tagline = generate_title_and_tagline(description)
    image = generate_image(description)
    poster = overlay_text(image, title, tagline)

    file_name = f"poster_{uuid.uuid4().hex}.png"
    save_path = os.path.join(OUTPUT_DIR, file_name)
    poster.save(save_path)
    return save_path
