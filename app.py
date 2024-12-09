import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
import requests
from copy import deepcopy
import cv2

import os
import math
import argparse
import random
import logging
from diffusers import StableDiffusionInpaintPipeline

import base64
import gradio as gr

from scipy.ndimage import zoom

import matplotlib.pyplot as plt
from demo import EditGuard_Hide, EditGuard_Reveal
from model_invert import *
import config as c

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision

from wam.watermark_anything.data.metrics import msg_predict_inference
from wam.notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img
)


def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = img_to_base64("logo.png")

html_content = f"""
<div style='display: flex; align-items: center; justify-content: center; padding: 20px;'>
    <img src='data:image/png;base64,{logo_base64}' alt='Logo' style='height: 50px; margin-right: 20px;'>
    <strong><font size='8'>EditGuard<font></strong>
</div>
"""

# Examples
examples = [
    ["examples/0000.png"],
    ["examples/0001.png"],
    ["examples/0011.png"],
    ["examples/0012.png"],
    ["examples/0014.png"],
    ["examples/0002.png"],
    ["examples/0003.png"],
    ["examples/0005.png"],
    ["examples/0006.png"],
    ["examples/0007.png"],
    ["examples/0008.png"],
    ["examples/0009.png"],
]

default_example = examples[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2",
                torch_dtype=torch.float16,
            ).to(device)

def load(net, name):
    state_dicts = torch.load(name)
    for k, v in state_dicts['net'].items():
        print(k)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if ('tmp_var' not in k) and ('bm' not in k)}
    net.load_state_dict(network_state_dict, strict=False)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def hiding(image_input, model):

    cover = np.array(image_input) / 255.
    cover = torch.from_numpy(cover).permute(2,0,1).unsqueeze(0).float().to(device)

    container = EditGuard_Hide(model, cover)

    return container, container

def WAM_hiding(image_input, model):

    img_pt = default_transform(image_input).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Embed the watermark message into the image
    wm_msg = model.get_random_msg(1)
    outputs = model.embed(img_pt, wm_msg)

    img_w = outputs['imgs_w']

    container =  unnormalize_img(img_w)
    print(container.min(), container.max())
    container = container.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
    container = np.clip(container, 0., 1.)
    container = Image.fromarray((container * 255).astype(np.uint8))

    # container.save("temp1.png")
    

    return container, container

def WAM_revealing(image_input, model_list, model):

    image_input = image_input / 255.
    image_input = torch.from_numpy(image_input).permute(2,0,1).unsqueeze(0).float().to(device)

    preds = model.detect(image_input)["preds"]  # [1, 33, 256, 256]
    mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
    bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits
    
    # Predict the embedded message and calculate bit accuracy
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()  # [1, 32]

    # Save the watermarked image and the detection mask
    
    mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(image_input.shape[-2], image_input.shape[-1]), mode="bilinear", align_corners=False)  # [1, 1, H, W]
    save_image(mask_preds_res, f"mask_pred.png")

    mask = mask_preds_res.permute(0, 2, 3, 1).cpu().squeeze().detach().numpy() * 255
    mask = 255. - mask
    mask = Image.fromarray(mask.astype(np.uint8))

    return mask

def rand(num_bits=100):
    random_str = ''.join([str(random.randint(0, 1)) for _ in range(num_bits)])
    return random_str

def ImageEdit(img, prompt, model_index):
    image, mask = img["image"], np.float32(img["mask"])

    h, w, c = image.shape
    prompt = ""
    image_init = Image.fromarray(image.astype(np.uint8), mode = "RGB")
    mask = Image.fromarray(mask.astype(np.uint8), mode = "RGB")
    image_inpaint = pipe(prompt=prompt, image=image_init, mask_image=mask, height=h, width=w).images[0]
    image_inpaint = np.array(image_inpaint) / 255.
    image = image / 255.
    mask_image = np.array(mask) / 255.
    mask_image = mask_image.astype(np.uint8)
    received_image = image * (1 - mask_image) + image_inpaint * mask_image

    return received_image, received_image, received_image


def image_model_select(ckp_index=0):
    net = Model()
    net.cuda()
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    load(net, c.MODEL_PATH + c.suffix)
    net.eval()
    
    return net

def wam_model_select(ckp_index=0):
    exp_dir = "/data03/zxy/OmniGuard/wam/checkpoints"
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
    wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    print("zxy", wam)
    
    return wam

# def revealing(image_edited, input_bit, model_list, model):

#     steg = image_edited / 255.
#     steg = torch.from_numpy(steg).permute(2,0,1).unsqueeze(0).float().to(device)
    
#     mask, remesg = EditGuard_Reveal(model, steg)
#     mask = mask.permute(0, 2, 3, 1).cpu().squeeze().numpy() * 255
#     mask = Image.fromarray(mask.astype(np.uint8))

#     remesg = (remesg > 0).int()
#     binary_string = ''.join(str(x.item()) for x in remesg.flatten())
    
#     # bit_acc = 1 - calculate_ber(input_bit, binary_string)

#     return mask

def revealing(image_edited, model_list, model):

    steg = image_edited / 255.
    steg = torch.from_numpy(steg).permute(2,0,1).unsqueeze(0).float().to(device)
    
    mask = EditGuard_Reveal(model, steg)
    mask = mask.permute(0, 2, 3, 1).cpu().squeeze().numpy() * 255
    mask = Image.fromarray(mask.astype(np.uint8))

    return mask


def equalize_string_lengths(bin_str1, bin_str2):
    # 获取两个字符串的长度
    len1 = len(bin_str1)
    len2 = len(bin_str2)
    
    # 比较长度并根据需要填充'0'
    if len1 < len2:
        # 对bin_str1进行填充
        bin_str1 = bin_str1.zfill(len2)
    elif len2 < len1:
        # 对bin_str2进行填充
        bin_str2 = bin_str2.zfill(len1)
    
    return bin_str1, bin_str2

def calculate_ber(string1, string2):

    string1, string2 = equalize_string_lengths(string1, string2)
    # 检查两个字符串长度是否相等
    if len(string1) != len(string2):
        raise ValueError("两个字符串长度必须相等")
    
    # 检查字符串是否只包含0和1
    if not all(c in '01' for c in string1 + string2):
        raise ValueError("字符串只能包含0和1")
    
    # 计算错位数
    errors = sum(1 for a, b in zip(string1, string2) if a != b)
    
    # 计算总位数
    total_bits = len(string1)
    
    # 计算误码率
    ber = errors / total_bits
    
    return ber


# Description
title = "<center><strong><font size='8'>EditGuard<font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

with gr.Blocks(css=css, title="EditGuard") as demo:
    gr.HTML(html_content)
    model = gr.State(value = None)
    model_wam = gr.State(value = None)
    save_h = gr.State(value = None)
    save_w = gr.State(value = None)
    sam_global_points = gr.State([])
    sam_global_point_label = gr.State([])
    sam_original_image = gr.State(value=None)
    sam_mask = gr.State(value=None)

    with gr.Tabs():
        with gr.TabItem('OmniGuard'):

            DESCRIPTION = """
            ## 使用方法：
            - 上传图像，点击"嵌入水印"按钮，生成带水印的图像。
            - 涂抹要编辑的区域，并使用Inpainting算法编辑图像。
            - 点击"提取"按钮检测篡改区域。"""
            
            gr.Markdown(DESCRIPTION)
            save_inpainted_image = gr.State(value=None)
            with gr.Column():
                with gr.Row():
                    model_list = gr.Dropdown(label="选择模型", choices=["模型1"], type = 'index')
                    clear_button = gr.Button("清除全部")
                with gr.Box():
                    gr.Markdown("# 1. 嵌入水印")
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(source='upload', label="原始图片", interactive=True, type="numpy", value=default_example[0])
                            # with gr.Row():
                                # bit_input = gr.Textbox(label="输入版权水印（100位比特序列）", placeholder="在这里输入...")
                                # rand_bit = gr.Button("🎲 随机生成版权水印")
                            hiding_button = gr.Button("嵌入水印")
                        with gr.Column():
                            image_watermark = gr.Image(source="upload", label="带有水印的图片", interactive=True, type="numpy")


                with gr.Box():
                    gr.Markdown("# 2. 篡改图片")
                    with gr.Row():
                        with gr.Column():
                            image_edit = gr.Image(source='upload',tool="sketch", label="选取篡改区域", interactive=True, type="numpy")
                            inpainting_model_list = gr.Dropdown(label="选择篡改模型", choices=["模型1：SD_inpainting"], type = 'index')
                            text_prompt = gr.Textbox(label="篡改提示词")
                            inpainting_button = gr.Button("篡改图片")
                        with gr.Column():
                            image_edited = gr.Image(source="upload", label="篡改结果", interactive=True, type="numpy")
                

                with gr.Box():
                    gr.Markdown("# 3. 提取水印&篡改区域")
                    with gr.Row():
                        with gr.Column():
                            image_edited_1 = gr.Image(source="upload", label="待提取图片", interactive=True, type="numpy")
                            
                            revealing_button = gr.Button("提取")
                        with gr.Column():
                            edit_mask = gr.Image(source='upload', label="编辑区域蒙版预测", interactive=True, type="numpy")
                            # bit_output = gr.Textbox(label="版权水印预测")
                            # acc_output = gr.Textbox(label="水印预测准确率")
                
                gr.Examples(
                            examples=examples,
                            inputs=[image_input],
                        )


                model_list.change(
                    image_model_select, inputs = [model_list], outputs=[model]
                    )
                hiding_button.click(
                    hiding, inputs=[image_input, model], outputs=[image_watermark, image_edit]
                    )
                # rand_bit.click(
                #     rand, inputs=[], outputs=[bit_input]
                #     )


                inpainting_button.click(
                    ImageEdit, inputs = [image_edit, text_prompt, inpainting_model_list], outputs=[image_edited, image_edited_1, save_inpainted_image]
                    )

                # revealing_button.click(
                #     revealing, inputs=[image_edited_1, bit_input, model_list, model], outputs=[edit_mask, bit_output, acc_output]
                #     )

                revealing_button.click(
                    revealing, inputs=[image_edited_1, model_list, model], outputs=[edit_mask]
                    )
        
        
        
        with gr.TabItem('Watermark Anything'):

            DESCRIPTION = """
            ## 使用方法：
            - 上传图像，点击"嵌入水印"按钮，生成带水印的图像。
            - 涂抹要编辑的区域，并使用Inpainting算法编辑图像。
            - 点击"提取"按钮检测篡改区域。"""
            
            gr.Markdown(DESCRIPTION)
            save_inpainted_image_wam = gr.State(value=None)
            with gr.Column():
                with gr.Row():
                    model_list_wam = gr.Dropdown(label="选择模型", choices=["模型1"], type = 'index')
                    clear_button_wam = gr.Button("清除全部")
                with gr.Box():
                    gr.Markdown("# 1. 嵌入水印")
                    with gr.Row():
                        with gr.Column():
                            image_input_wam = gr.Image(source='upload', label="原始图片", interactive=True, type="numpy", value=default_example[0])
                            # with gr.Row():
                                # bit_input = gr.Textbox(label="输入版权水印（100位比特序列）", placeholder="在这里输入...")
                                # rand_bit = gr.Button("🎲 随机生成版权水印")
                            hiding_button_wam = gr.Button("嵌入水印")
                        with gr.Column():
                            image_watermark_wam = gr.Image(source="upload", label="带有水印的图片", interactive=True, type="numpy")


                with gr.Box():
                    gr.Markdown("# 2. 篡改图片")
                    with gr.Row():
                        with gr.Column():
                            image_edit_wam = gr.Image(source='upload',tool="sketch", label="选取篡改区域", interactive=True, type="numpy")
                            inpainting_model_list_wam = gr.Dropdown(label="选择篡改模型", choices=["模型1：SD_inpainting"], type = 'index')
                            text_prompt_wam = gr.Textbox(label="篡改提示词")
                            inpainting_button_wam = gr.Button("篡改图片")
                        with gr.Column():
                            image_edited_wam = gr.Image(source="upload", label="篡改结果", interactive=True, type="numpy")
                

                with gr.Box():
                    gr.Markdown("# 3. 提取水印&篡改区域")
                    with gr.Row():
                        with gr.Column():
                            image_edited_1_wam = gr.Image(source="upload", label="待提取图片", interactive=True, type="numpy")
                            
                            revealing_button_wam = gr.Button("提取")
                        with gr.Column():
                            edit_mask_wam = gr.Image(source='upload', label="编辑区域蒙版预测", interactive=True, type="numpy")
                
                gr.Examples(
                            examples=examples,
                            inputs=[image_input],
                        )


                model_list_wam.change(
                    wam_model_select, inputs = [model_list_wam], outputs=[model_wam]
                    )
                hiding_button_wam.click(
                    WAM_hiding, inputs=[image_input_wam, model_wam], outputs=[image_watermark_wam, image_edit_wam]
                    )

                inpainting_button_wam.click(
                    ImageEdit, inputs = [image_edit_wam, text_prompt_wam, inpainting_model_list_wam], outputs=[image_edited_wam, image_edited_1_wam, save_inpainted_image_wam]
                    )

                revealing_button_wam.click(
                    WAM_revealing, inputs=[image_edited_1_wam, model_list_wam, model_wam], outputs=[edit_mask_wam]
                    )


demo.launch(server_name="0.0.0.0", server_port=10049, share=True, favicon_path='logo.png')

