---
license: other
license_name: bespoke-lora-trained-license
license_link: https://multimodal.art/civitai-licenses?allowNoCredit=True&allowCommercialUse=Image&allowDerivatives=False&allowDifferentLicense=False
tags:
- text-to-image
- stable-diffusion
- lora
- diffusers
- template:sd-lora
- migrated
- movie
- style
- poster
- movie poster

base_model: black-forest-labs/FLUX.1-dev
instance_prompt: mvpstrCE style
widget:
- text: 'Display the text "Movie Poster" on a movie poster.

mvpstrCE style'
  
  output:
    url: >-
      27819646.jpeg
- text: '"Fake Life".

Ethereal imagery.

mvpstrCE style'
  
  output:
    url: >-
      27808030.jpeg
- text: '"Rule Breaker".

mvpstrCE style'
  
  output:
    url: >-
      27808025.jpeg
- text: '"Demon Seed".

mvpstrCE style'
  
  output:
    url: >-
      27808028.jpeg
- text: '"Hidden Evil".

mvpstrCE style'
  
  output:
    url: >-
      27808019.jpeg
- text: '"True Lies".

mvpstrCE style'
  
  output:
    url: >-
      27808016.jpeg
- text: '"The Four Seasons".

Nature scenery.

mvpstrCE style'
  
  output:
    url: >-
      27808029.jpeg
- text: 'Title: "Area 51". Cute fluffy aliens in a science lab.

Subtext: "It''s real!".

Show movie credits.

mvpstrCE style'
  
  output:
    url: >-
      31341915.jpeg
- text: '"Ether Waste". Lettering atop a chaotic and abstract jumble of background 3d letters, overlapping and jumbled across the page.

Subtitle: "The end is near". Written By "Patrick Lee".

mvpstrCE style'
  
  output:
    url: >-
      32833689.jpeg
- text: 'Horror movie poster. Bloody title: "Orphan".
Crimson subtitle: "HIDE THE KNIVES".

Holding a bloody knife behind her back, viewed from behind.

A young girl with pigtails, holding a bloody knife. 

Atmospheric.

mvpstrCE style'
  
  output:
    url: >-
      37643308.jpeg
- text: 'Movie Poster. Bloody Title: "THE BAD SEED". Subtitle: "EVIL IS SOMETIMES DISGUISED". 

a girl holding a knife behind her back. She has pigtails and stands on a dock, hiding a knife behind her back. Viewed from behind.

atmospheric, foggy, spooky. Nighttime. Full moon. 

Soft watercolour. 

mvpstrCE style'
  
  output:
    url: >-
      37643306.jpeg

---

# Movie Poster - CE - SDXL & Flux 

<Gallery />



([CivitAI](https://civitai.com/models/))

## Model description

<p><strong>Please share your creations. I would love to see how you have used the LoRA.</strong></p><p>Weight around 1.0  suits most pictures.</p><p>Please do not distribute this LoRA to other platforms. I will be very cranky if I discover it elsewhere.</p><p>Whilst I would appreciate credit in your images, I am not insisting on it.</p>

## Trigger words
You should use `mvpstrCE style` to trigger the image generation.
    

## Download model

Weights for this model are available in Safetensors format.

[Download](/Keltezaa/movie-poster-ce-sdxl-flux/tree/main) them in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16).to(device)
pipeline.load_lora_weights('Keltezaa/movie-poster-ce-sdxl-flux', weight_name='MoviePoster03-02_CE_FLUX_128AIT.safetensors')
image = pipeline('Movie Poster. Bloody Title: "THE BAD SEED". Subtitle: "EVIL IS SOMETIMES DISGUISED". 

a girl holding a knife behind her back. She has pigtails and stands on a dock, hiding a knife behind her back. Viewed from behind.

atmospheric, foggy, spooky. Nighttime. Full moon. 

Soft watercolour. 

mvpstrCE style').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

