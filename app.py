#!/usr/bin/env python3
import gradio as gr
import os
from clip_interrogator import Config, Interrogator
from huggingface_hub import hf_hub_download
from share_btn import community_icon_html, loading_icon_html, share_js

MODELS = ['ViT-L (best for Stable Diffusion 1.*)', 'ViT-H (best for Stable Diffusion 2.*)']

# download preprocessed files
PREPROCESS_FILES = [
    'ViT-H-14_laion2b_s32b_b79k_artists.pkl',
    'ViT-H-14_laion2b_s32b_b79k_flavors.pkl',
    'ViT-H-14_laion2b_s32b_b79k_mediums.pkl',
    'ViT-H-14_laion2b_s32b_b79k_movements.pkl',
    'ViT-H-14_laion2b_s32b_b79k_trendings.pkl',
    'ViT-L-14_openai_artists.pkl',
    'ViT-L-14_openai_flavors.pkl',
    'ViT-L-14_openai_mediums.pkl',
    'ViT-L-14_openai_movements.pkl',
    'ViT-L-14_openai_trendings.pkl',
]
print("Download preprocessed cache files...")
for file in PREPROCESS_FILES:
    path = hf_hub_download(repo_id="pharma/ci-preprocess", filename=file, cache_dir="cache")
    cache_path = os.path.dirname(path)


# load BLIP and ViT-L 
config = Config(cache_path=cache_path, clip_model_path="cache", clip_model_name="ViT-L-14/openai")
ci_vitl = Interrogator(config)
ci_vitl.clip_model = ci_vitl.clip_model.to("cpu")

# load ViT-H
config.blip_model = ci_vitl.blip_model
config.clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
ci_vith = Interrogator(config)
ci_vith.clip_model = ci_vith.clip_model.to("cpu")


def inference(image, clip_model_name, mode):

    # move selected model to GPU and other model to CPU
    if clip_model_name == MODELS[0]:
        ci_vith.clip_model = ci_vith.clip_model.to("cpu")
        ci_vitl.clip_model = ci_vitl.clip_model.to(ci_vitl.device)
        ci = ci_vitl
    else:
        ci_vitl.clip_model = ci_vitl.clip_model.to("cpu")
        ci_vith.clip_model = ci_vith.clip_model.to(ci_vith.device)
        ci = ci_vith

    ci.config.blip_num_beams = 64
    ci.config.chunk_size = 2048
    ci.config.flavor_intermediate_count = 2048 if clip_model_name == MODELS[0] else 1024

    image = image.convert('RGB')
    if mode == 'best':
        prompt = ci.interrogate(image)
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image)
    else:
        prompt = ci.interrogate_fast(image)

    return prompt, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


TITLE = """
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

ARTICLE = """
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
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
</div>
"""

CSS = '''
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
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
#share-btn-container .wrap {
    display: none !important;
}
'''

with gr.Blocks(css=CSS) as block:
    with gr.Column(elem_id="col-container"):
        gr.HTML(TITLE)

        input_image = gr.Image(type='pil', elem_id="input-img")
        input_model = gr.Dropdown(MODELS, value=MODELS[0], label='CLIP Model')
        input_mode = gr.Radio(['best', 'fast'], value='best', label='Mode')
        submit_btn = gr.Button("Submit")
        output_text = gr.Textbox(label="Output", elem_id="output-txt")

        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)

        examples=[['example01.jpg', MODELS[0], 'best'], ['example02.jpg', MODELS[0], 'best']]
        ex = gr.Examples(
            examples=examples, 
            fn=inference, 
            inputs=[input_image, input_model, input_mode], 
            outputs=[output_text, share_button, community_icon, loading_icon], 
            cache_examples=True, 
            run_on_click=True
        )
        ex.dataset.headers = [""]
        
        gr.HTML(ARTICLE)

    submit_btn.click(
        fn=inference, 
        inputs=[input_image, input_model, input_mode], 
        outputs=[output_text, share_button, community_icon, loading_icon]
    )
    share_button.click(None, [], [], _js=share_js)

block.queue(max_size=32).launch(show_api=False)
