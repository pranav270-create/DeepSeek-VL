# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to PERMIT persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM
import time
import os


global device
dtype = torch.float16
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.bfloat16
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images, load_pil_images_plain

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(dtype).to(device).eval()

# single image conversation example
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Please classify each document as either a 1) ZONING MAP, 2) LEGAL DOC, 3) PERMIT, or 4) OTHER. Do not include other text, just the classification label.",
        "images": ["./images/training_pipelines.jpg"],
    },
    {"role": "Assistant", "content": ""},
]

# multiple images (or in-context learning) conversation example
# conversation = [
#     {
#         "role": "User",
#         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
#                    "<image_placeholder>a dog wearing a santa hat, "
#                    "<image_placeholder>a dog wearing a wizard outfit, and "
#                    "<image_placeholder>what's the dog wearing?",
#         "images": [
#             "images/dog_a.png",
#             "images/dog_b.png",
#             "images/dog_c.png",
#             "images/dog_d.png",
#         ],
#     },
#     {"role": "Assistant", "content": ""}
# ]

# load images and prepare for inputs
# pil_images = load_pil_images(conversation)
# prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device, dtype=torch.float32)

user_prompt = "<image_placeholder>Please classify each document as either a 1) ZONING MAP, 2) LEGAL DOC, 3) PERMIT, or 4) OTHER. Do not include other text, just the classification label."
user_prompt = "<image_placeholder>Please describe each document in the image in detail"
batch_images = load_pil_images_plain(["./images/dog_a.png", "./images/dog_b.png", "./images/dog_c.png", "./images/dog_d.png"])
batch_conversation = [[
        {
            "role": "User",
            "content": user_prompt,
            "images": ["./images/Image1.png"],
        },
        {"role": "Assistant", "content": ""},
    ],
    [
        {
            "role": "User",
            "content": user_prompt,
            "images": ["./images/Image2.png"],
        },
        {"role": "Assistant", "content": ""},
    ],
    [
        {
            "role": "User",
            "content": user_prompt,
            "images": ["./images/Image3.png"],
        },
        {"role": "Assistant", "content": ""},
    ],
    [
        {
            "role": "User",
            "content": user_prompt,
            "images": ["./images/Image4.png"],
        },
        {"role": "Assistant", "content": ""},
    ],
]

start = time.time()
# prepare the inputs
prepare_batch_inputs = vl_chat_processor.batch_call(conversations=batch_conversation, images=batch_images, force_batchify=True, dtype=dtype).to(vl_gpt.device)
print(time.time() - start)

# get the inputs embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_batch_inputs)
print(time.time() - start)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_batch_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=200,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(time.time() - start)
print(answer)
