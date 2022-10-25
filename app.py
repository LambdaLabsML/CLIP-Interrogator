import sys
sys.path.append('src/blip')
sys.path.append('src/clip')

import clip
import gradio as gr
import hashlib
import math
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.blip import blip_decoder
from PIL import Image
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from share_btn import community_icon_html, loading_icon_html, share_js

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading BLIP model...")
blip_image_eval_size = 384
blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'        
blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit='large', med_config='./src/blip/configs/med_config.json')
blip_model.eval()
blip_model = blip_model.to(device)

print("Loading CLIP model...")
clip_model_name = 'ViT-L/14' # https://huggingface.co/openai/clip-vit-large-patch14
clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
clip_model.to(device).eval()

chunk_size = 2048
flavor_intermediate_count = 2048


class LabelTable():
    def __init__(self, labels, desc):
        self.labels = labels
        self.embeds = []

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        os.makedirs('./cache', exist_ok=True)
        cache_filepath = f"./cache/{desc}.pkl"
        if desc is not None and os.path.exists(cache_filepath):
            with open(cache_filepath, 'rb') as f:
                data = pickle.load(f)
                if data['hash'] == hash:
                    self.labels = data['labels']
                    self.embeds = data['embeds']

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None):
                text_tokens = clip.tokenize(chunk).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            with open(cache_filepath, 'wb') as f:
                pickle.dump({"labels":self.labels, "embeds":self.embeds, "hash":hash}, f)
    
    def _rank(self, image_features, text_embeds, top_count=1):
        top_count = min(top_count, len(text_embeds))
        similarity = torch.zeros((1, len(text_embeds))).to(device)
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).float().to(device)
        for i in range(image_features.shape[0]):
            similarity += (image_features[i].unsqueeze(0) @ text_embeds.T).softmax(dim=-1)
        _, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features, top_count=1):
        if len(self.labels) <= chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/chunk_size))
        keep_per_chunk = int(chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks)):
            start = chunk_idx*chunk_size
            stop = min(start+chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]

def generate_caption(pil_image):
    gpu_image = T.Compose([
        T.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=TF.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]

def load_list(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def rank_top(image_features, text_array):
    text_tokens = clip.tokenize([text for text in text_array]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = torch.zeros((1, len(text_array)), device=device)
    for i in range(image_features.shape[0]):
        similarity += (image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)

    _, top_labels = similarity.cpu().topk(1, dim=-1)
    return text_array[top_labels[0][0].numpy()]

def similarity(image_features, text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()       
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity[0][0]

def interrogate(image):
    caption = generate_caption(image)

    images = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

    flaves = flavors.rank(image_features, flavor_intermediate_count)
    best_medium = mediums.rank(image_features, 1)[0]
    best_artist = artists.rank(image_features, 1)[0]
    best_trending = trendings.rank(image_features, 1)[0]
    best_movement = movements.rank(image_features, 1)[0]

    best_prompt = caption
    best_sim = similarity(image_features, best_prompt)

    def check(addition):
        nonlocal best_prompt, best_sim
        prompt = best_prompt + ", " + addition
        sim = similarity(image_features, prompt)
        if sim > best_sim:
            best_sim = sim
            best_prompt = prompt
            return True
        return False

    def check_multi_batch(opts):
        nonlocal best_prompt, best_sim
        prompts = []
        for i in range(2**len(opts)):
            prompt = best_prompt
            for bit in range(len(opts)):
                if i & (1 << bit):
                    prompt += ", " + opts[bit]
            prompts.append(prompt)

        prompt = rank_top(image_features, prompts)
        sim = similarity(image_features, prompt)
        if sim > best_sim:
            best_sim = sim
            best_prompt = prompt

    check_multi_batch([best_medium, best_artist, best_trending, best_movement])

    extended_flavors = set(flaves)
    for _ in tqdm(range(25), desc="Flavor chain"):
        try:
            best = rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
            flave = best[len(best_prompt)+2:]
            if not check(flave):
                break
            extended_flavors.remove(flave)
        except:
            # exceeded max prompt length
            break

    return best_prompt


sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
trending_list = [site for site in sites]
trending_list.extend(["trending on "+site for site in sites])
trending_list.extend(["featured on "+site for site in sites])
trending_list.extend([site+" contest winner" for site in sites])

raw_artists = load_list('data/artists.txt')
artists = [f"by {a}" for a in raw_artists]
artists.extend([f"inspired by {a}" for a in raw_artists])

artists = LabelTable(artists, "artists")
flavors = LabelTable(load_list('data/flavors.txt'), "flavors")
mediums = LabelTable(load_list('data/mediums.txt'), "mediums")
movements = LabelTable(load_list('data/movements.txt'), "movements")
trendings = LabelTable(trending_list, "trendings")


def inference(image):
    return interrogate(image), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

title = """
    <div style="text-align: center; max-width: 650px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px;">
            CLIP Interrogator
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Want to figure out what a good prompt might be to create new images like an existing one? The CLIP Interrogator is here to get you answers!
        </p>
    </div>
"""
article = """
<p>
Example art by <a href="https://pixabay.com/illustrations/watercolour-painting-art-effect-4799014/">Layers</a> 
and <a href="https://pixabay.com/illustrations/animal-painting-cat-feline-pet-7154059/">Lin Tong</a> 
from pixabay.com
</p>

<p>
Server busy? You can also run on <a href="https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb">Google Colab</a>
</p>

<p>
Has this been helpful to you? Follow me on twitter 
<a href="https://twitter.com/pharmapsychotic">@pharmapsychotic</a> 
and check out more tools at my
<a href="https://pharmapsychotic.com/tools.html">Ai generative art tools list</a>
</p>
"""

css = '''
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container wrap {
    display: none !important;
}
'''

with gr.Blocks(css=css) as block:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)

        input_image = gr.Image(type='pil', elem_id="input-img")
        submit_btn = gr.Button("Submit")
        output_text = gr.Textbox(label="Output", elem_id="output-txt")

        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)

        examples=[['example01.jpg'], ['example02.jpg']]
        ex = gr.Examples(examples=examples, fn=inference, inputs=input_image, outputs=[output_text, share_button, community_icon, loading_icon], cache_examples=True, run_on_click=True)
        ex.dataset.headers = [""]
        
        gr.HTML(article)

    submit_btn.click(fn=inference, inputs=input_image, outputs=[output_text, share_button, community_icon, loading_icon])
    share_button.click(None, [], [], _js=share_js)

block.queue(max_size=32).launch(show_api=False)
