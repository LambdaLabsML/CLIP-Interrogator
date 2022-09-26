import gradio as gr
import sys
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.append('src/blip')
sys.path.append('src/clip')

import clip
from models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading BLIP model...")
blip_image_eval_size = 384
blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'        
blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit='large', med_config='./src/blip/configs/med_config.json')
blip_model.eval()
blip_model = blip_model.to(device)

print("Loading CLIP model...")
clip_model_name = 'ViT-L/14'
clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
clip_model.to(device).eval()


def generate_caption(pil_image):
    gpu_image = T.Compose([
        T.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=TF.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]

def inference(image):
    return generate_caption(image)
    
inputs = [gr.inputs.Image(type='pil')]
outputs = gr.outputs.Textbox(label="Output")

title = "CLIP Interrogator"
description = "First test of CLIP Interrogator on HuggingSpace"
article = """
<p style='text-align: center'>
    <a href="">Colab Notebook</a> /
    <a href="">Github repo</a>
</p>
"""

gr.Interface(
    inference, 
    inputs, 
    outputs, 
    title=title, description=description, 
    article=article, 
    examples=[['example.jpg']]
).launch(enable_queue=True)
