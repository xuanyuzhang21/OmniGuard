import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from jpegtest import JpegTest
import torchvision

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the specified checkpoint
exp_dir = "checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

# Define the directory containing the images to watermark
img_dir = "assets/images"  # Directory containing the original images
output_dir = "outputs"  # Directory to save the watermarked images
os.makedirs(output_dir, exist_ok=True)
# Define a 32-bit message to be embedded into the images
# wm_msg = torch.randint(0, 2, (32,)).float().to(device)
wm_msg = wam.get_random_msg(1)  # [1, 32]

# Proportion of the image to be watermarked (0.5 means 50% of the image).
# This is used here to show the watermark localization property. In practice, you may want to use a predifined mask or the entire image.
# proportion_masked = 0.5

# Iterate over each image in the directory
for img_ in os.listdir(img_dir):
    # Load and preprocess the image
    img_path = os.path.join(img_dir, img_)
    img = Image.open(img_path).convert("RGB")
    img_pt = default_transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Embed the watermark message into the image
    outputs = wam.embed(img_pt, wm_msg)

    # Create a random mask to watermark only a part of the image
    # mask = create_random_mask(img_pt, num_masks=1,mask_percentage=proportion_masked)  # [1, 1, H, W]
    # img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)  # [1, 3, H, W]
    img_w = outputs['imgs_w']

    # img_w = (img_w - img_w.min()) / (img_w.max() - img_w.min())
    # img_w = img_w.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
    # img_w = Image.fromarray((img_w * 255).astype(np.uint8))
    # img_w = img_w.save("temp.png", format="JPEG", quality=70, subsampling=0)

    # NL = 70
    # NL = int(np.random.randint(70, 85))
    # DiffJPEG_opt = JpegTest(NL, 0)
    # img_w = DiffJPEG_opt(img_w)
    # img_w = Image.open("temp.png").convert("RGB")
    # img_w = default_transform(img_w).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Detect the watermark in the watermarked image
    preds = wam.detect(img_w)["preds"]  # [1, 33, 256, 256]
    mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
    bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits
    
    # Predict the embedded message and calculate bit accuracy
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()  # [1, 32]
    bit_acc = (pred_message == wm_msg).float().mean().item()

    # Save the watermarked image and the detection mask
    mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)  # [1, 1, H, W]
    save_image(unnormalize_img(img_w), f"{output_dir}/{img_}_wm.png")
    save_image(mask_preds_res, f"{output_dir}/{img_}_pred.png")
    # save_image(mask, f"{output_dir}/{img_}_target.png")
    
    # Print the predicted message and bit accuracy for each image
    print(f"Predicted message for image {img_}: ", pred_message[0].numpy())
    print(f"Bit accuracy for image {img_}: ", bit_acc)