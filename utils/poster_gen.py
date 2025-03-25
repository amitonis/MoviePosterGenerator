import torch  # type: ignore
from diffusers import StableDiffusionPipeline  # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import os
import uuid
import textwrap

# Load the models once
pipe = StableDiffusionPipeline.from_pretrained(
    "C:/Users/amitm/.cache/huggingface/hub/models--ogkalu--Comic-Diffusion/snapshots/ff684f581ab24e094e2055d9422e9ee076d139a8",
    torch_dtype=torch.float16
)
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
    # Generate higher-resolution poster (portrait format)
    image = pipe(description, height=896, width=640).images[0]
    return image

def overlay_text(image, title, tagline):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    max_title_font_size = 60
    max_tagline_font_size = 32

    # Utility: wrap and resize text to fit image
    def draw_wrapped(draw, text, font_path, max_font_size, max_width, y, fill="white", align="center"):
        if not text.strip():  # ✅ Skip empty text
            return

        for font_size in range(max_font_size, 10, -2):
            font = ImageFont.truetype(font_path, font_size)
            lines = textwrap.wrap(text, width=25)
            if lines:
                max_line_width = max([(font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines])
                if max_line_width <= max_width:
                    break

        total_height = sum([(font.getbbox(line)[3] - font.getbbox(line)[1]) for line in lines]) + len(lines) * 5
        y_start = y

        for line in lines:
            line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
            x = (image.width - line_width) / 2
            draw.text((x, y_start), line, font=font, fill=fill)
            y_start += font.getbbox(line)[3] - font.getbbox(line)[1] + 5

    # Draw title at top
    draw_wrapped(draw, title, FONT_PATH_TITLE, max_title_font_size, width * 0.9, y=40)

    # Draw tagline at bottom
    draw_wrapped(draw, tagline, FONT_PATH_TAGLINE, max_tagline_font_size, width * 0.9, y=height - 150)

    return image

def generate_poster(description):
    title, tagline = generate_title_and_tagline(description)
    image = generate_image(description)
    poster = overlay_text(image, title, tagline)

    file_name = f"poster_{uuid.uuid4().hex}.png"
    save_path = os.path.join(OUTPUT_DIR, file_name)
    poster.save(save_path)
    return save_path
