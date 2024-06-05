import json
import random
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from PIL import Image

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
from segment_anything import build_sam, SamPredictor, build_sam_hq
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

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def write_masks_to_png(masks, image, path: str) -> None:
    plt.figure()
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    #plt.show()
    plt.savefig(path)
    return



app = Flask(__name__)
CORS(app)

config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "groundingdino_swint_ogc.pth"
sam_checkpoint = "sam_hq_vit_h.pth"
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
    if "seg" in name:
        image_folder = 'seg_results'
    elif "combined" in name:
        image_folder = 'outputs/masks'
    else:
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

    image_file1 = request.files['img']
    image_file2 = request.files['example']

    device = "cuda"
    sam_checkpoint = "sam_hq_vit_h.pth"

    if 'points' not in request.form:
        
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
    else:
        try:
            img_pil, img = load_image(image_file1)
            example_pil, example = load_image(image_file2)
        except IOError:
            return "Error: Unable to open one of the images.", 400

        points = request.form['points']
        points = json.loads(points)

        labels = request.form['labels']
        labels = json.loads(labels)
        
        input_points = []

        for p in points:
            input_points.append([p['x']*img_pil.size[0], p['y']*img_pil.size[1]])
        input_points = np.array(input_points)
        labels = np.array(labels)
        print(input_points)

        sam_checkpoint = "sam_hq_vit_h.pth"
        model_type = "vit_h"

        device = "cuda"

        predictor = SamPredictor(build_sam_hq(checkpoint=sam_checkpoint).to(device))
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)

        # predictor = SamPredictor(sam)

        image = np.array(img_pil)
        # Convert RGB to BGR
        image = image[:, :, ::-1].copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        masks, _, _ = predictor.predict(
            point_coords = input_points,
            point_labels = labels,
            multimask_output = False,
        )

        mask = masks[0]
        mask_pil = Image.fromarray(mask)

        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        maskname = "combined-"+timestamp+".jpg"

        mask_pil.save("outputs/masks/maskWithPoints.jpg")
        combined = apply_mask(img_pil, mask_pil)
        combined = combined.convert("RGB")
        combined.save("outputs/masks/"+maskname)

        # ----------------start painting--------------------------
        size = img_pil.size
        H, W = size[1], size[0]

        size = (600, 600*H//W//8*8)

        image_pil = img_pil.resize(size)
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

        return {"filename": filename, "maskedImg": maskname}, 200


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def apply_mask(main_image, mask_image, opacity=64):
    # Ensure the images are in the same size
    mask_image = mask_image.resize(main_image.size)

    # Convert mask to RGBA if not already
    if mask_image.mode != 'RGBA':
        mask_image = mask_image.convert('RGBA')

    # Generate one random color for the whole mask
    color = random_color()

    # Create a new image for the overlay with the random color
    overlay = Image.new('RGBA', main_image.size, (0, 0, 0, 0))  # Initialize to transparent

    # Apply the mask
    for x in range(main_image.size[0]):
        for y in range(main_image.size[1]):
            if mask_image.getpixel((x, y))[0] > 128:  # assuming mask is grayscale
                overlay.putpixel((x, y), (*color, opacity))

    # Blend with the main image
    combined = Image.alpha_composite(main_image.convert('RGBA'), overlay)
    return combined


@app.route('/sam', methods=['POST'])
def run_sam():
    if 'img' not in request.files:
        return "Missing images", 400
    
    print("running sam")
    image = request.files['img']

    CHECKPOINT_PATH='sam_vit_h_4b8939.pth'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)

    img_pil, img = load_image(image)

    image_rgb = cv2.cvtColor(np.array(img_pil)[:, :, ::-1].copy(), cv2.COLOR_BGR2RGB)
    # Generate segmentation mask
    output_mask = mask_generator.generate(image_rgb)

    # sam_masks = inpalib.generate_sam_masks(input_image, sam_model_id, anime_style_chk)
    # sam_masks = inpalib.sort_masks_by_area(sam_masks)
    # sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, sam_dict["pad_mask"])

    # seg_image = inpalib.create_seg_color_image(input_image, sam_masks)

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    filename = "seg_color_image_" + timestamp + ".png"

    write_masks_to_png(output_mask, image_rgb, "seg_results/" + filename)

    return {"filename": filename}, 200
    # inpalib = importlib.import_module("inpaint-anything.inpalib")

    # # available_sam_ids = inpalib.get_available_sam_ids()

    # use_sam_id = "sam_vit_h_4b8939.pth"
    # # assert use_sam_id in available_sam_ids, f"Invalid SAM ID: {use_sam_id}"
    # input_image = np.array(Image.open(image))

    # sam_masks = inpalib.generate_sam_masks(input_image, use_sam_id, anime_style_chk=False)
    # sam_masks = inpalib.sort_masks_by_area(sam_masks)

    # seg_color_image = inpalib.create_seg_color_image(input_image, sam_masks)

    
    
if __name__ == '__main__':
    app.run(debug=True)
