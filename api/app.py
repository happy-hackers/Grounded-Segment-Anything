from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import io

import os, time

import numpy as np
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# diffusers
from diffusers import PaintByExamplePipeline


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)




app = Flask(__name__)
CORS(app)

config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "groundingdino_swint_ogc.pth"
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda"
output_dir = "outputs"
det_prompt = "benchtop"
box_threshold = 0.3
text_threshold = 0.25
# make dir
os.makedirs(output_dir, exist_ok=True)
# load model
model = load_model(config_file, grounded_checkpoint, device=device)

@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"

@app.route('/file/<name>', methods=['GET'])
def getImage(name):
    # Set the path to the images folder
    image_folder = 'outputs'
    try:
        # This will send the requested file from the specified directory
        return send_from_directory(image_folder, name)
    except FileNotFoundError:
        return "File not found", 404

@app.route('/generate', methods=['POST'])
def paint():
    # Check if both images are received
    if 'img' not in request.files or 'example' not in request.files:
        return "Missing images", 400

    # Open both images
    image_file1 = request.files['img']
    image_file2 = request.files['example']
    try:
        img_pil, img = load_image(image_file1)
        example_pil, example = load_image(image_file2)
    except IOError:
        return "Error: Unable to open one of the images.", 400

    boxes_filt, pred_phrases = get_grounding_output(
        model, img, det_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    image = np.array(img_pil)
    # Convert RGB to BGR
    image = image[:, :, ::-1].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = img_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    plt.savefig(os.path.join("./outputs/masks/mask-" + timestamp + ".jpg"), bbox_inches="tight")

    # inpainting pipeline
    mask = masks[0][0].cpu().numpy()
    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)

    size = (600, 600*H//W//8*8)

    image_pil = image_pil.resize(size)
    mask_pil = mask_pil.resize(size)
    example_pil = example_pil.resize(size)

    pipe = PaintByExamplePipeline.from_pretrained(
        "Fantasy-Studio/Paint-by-Example",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    result = pipe(image=image_pil, mask_image=mask_pil, example_image=example_pil).images[0]

    
    filename = "output-" + timestamp + ".jpg"
    result.save("./outputs/" + filename)

    return {"filename": filename}, 200

if __name__ == '__main__':
    app.run(debug=True)