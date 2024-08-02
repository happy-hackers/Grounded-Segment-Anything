import json
import random
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from PIL import Image

import os, time

import numpy as np
import torch

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
from diffusers import PaintByExamplePipeline, StableDiffusionControlNetPipeline, ControlNetModel


# using lora + controlnet_depth
from transformers import pipeline
from diffusers import DPMSolverMultistepScheduler
# from diffusers.utils import load_image

from datetime import datetime
import urllib.request
import base64
import json
import time
import os

webui_server_url = 'http://127.0.0.1:7860'

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(**payload):
    response = call_api('sdapi/v1/txt2img', **payload)
    t = timestamp()
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join("outputs", f'lora-{t}-{index}.png')
        decode_and_save_base64(image, save_path)
    return "outputs/lora-" + t + "-0.png"

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
def generate_with_webui_api():
    payload = {"alwayson_scripts": {"API payload": {"args": []}, "Comments": {"args": []}, "ControlNet": {"args": [{"advanced_weighting": None, "animatediff_batch": False, "batch_image_files": [], "batch_images": "", "batch_keyframe_idx": None, "batch_mask_dir": None, "batch_modifiers": [], "control_mode": "Balanced", "effective_region_mask": None, "enabled": True, "guidance_end": 1.0, "guidance_start": 0.0, "hr_option": "Both", "image": {"image": encode_file_to_base64(r"/home/jupyter/Grounded-Segment-Anything/assets/stone/black-benchtop.jpeg"), "mask": None}, "inpaint_crop_input_image": False, "input_mode": "simple", "ipadapter_input": None, "is_ui": True, "loopback": False, "low_vram": False, "mask": None, "model": "control_sd15_depth [fef5e48e]", "module": "depth_midas", "output_dir": "", "pixel_perfect": False, "processor_res": 512, "pulid_mode": "Fidelity", "resize_mode": "Just Resize", "save_detected_map": True, "threshold_a": 0.5, "threshold_b": 0.5, "union_control_type": "Depth", "weight": 1.0}, {"advanced_weighting": None, "animatediff_batch": False, "batch_image_files": [], "batch_images": "", "batch_keyframe_idx": None, "batch_mask_dir": None, "batch_modifiers": [], "control_mode": "Balanced", "effective_region_mask": None, "enabled": False, "guidance_end": 1.0, "guidance_start": 0.0, "hr_option": "Both", "image": None, "inpaint_crop_input_image": False, "input_mode": "simple", "ipadapter_input": None, "is_ui": True, "loopback": False, "low_vram": False, "mask": None, "model": "None", "module": "none", "output_dir": "", "pixel_perfect": False, "processor_res": -1, "pulid_mode": "Fidelity", "resize_mode": "Crop and Resize", "save_detected_map": True, "threshold_a": -1.0, "threshold_b": -1.0, "union_control_type": "Unknown", "weight": 1.0}, {"advanced_weighting": None, "animatediff_batch": False, "batch_image_files": [], "batch_images": "", "batch_keyframe_idx": None, "batch_mask_dir": None, "batch_modifiers": [], "control_mode": "Balanced", "effective_region_mask": None, "enabled": False, "guidance_end": 1.0, "guidance_start": 0.0, "hr_option": "Both", "image": None, "inpaint_crop_input_image": False, "input_mode": "simple", "ipadapter_input": None, "is_ui": True, "loopback": False, "low_vram": False, "mask": None, "model": "None", "module": "none", "output_dir": "", "pixel_perfect": False, "processor_res": -1, "pulid_mode": "Fidelity", "resize_mode": "Crop and Resize", "save_detected_map": True, "threshold_a": -1.0, "threshold_b": -1.0, "union_control_type": "Unknown", "weight": 1.0}]}, "Extra options": {"args": []}, "Hypertile": {"args": []}, "Refiner": {"args": [False, "", 0.8]}, "Sampler": {"args": [20, "DPM++ 2M", "Automatic"]}, "Seed": {"args": [-1, False, -1, 0, 0, 0]}, "VRAM Usage Estimator": {"args": [""]}}, "batch_size": 3, "cfg_scale": 7, "comments": {}, "denoising_strength": 0.7, "disable_extra_networks": False, "do_not_save_grid": False, "do_not_save_samples": False, "enable_hr": False, "height": 800, "hr_negative_prompt": "", "hr_prompt": "", "hr_resize_x": 0, "hr_resize_y": 0, "hr_scale": 2, "hr_second_pass_steps": 0, "hr_upscaler": "Latent", "n_iter": 1, "negative_prompt": "white countertop", "override_settings": {}, "override_settings_restore_afterwards": True, "prompt": " <lora:black-marble-counter:1>, a kitchen with a black marble countertop", "restore_faces": False, "s_churn": 0.0, "s_min_uncond": 0.0, "s_noise": 1.0, "s_tmax": None, "s_tmin": 0.0, "sampler_name": "DPM++ 2M", "scheduler": "Automatic", "script_args": [], "script_name": None, "seed": -1, "seed_enable_extras": True, "seed_resize_from_h": -1, "seed_resize_from_w": -1, "steps": 20, "styles": [], "subseed": -1, "subseed_strength": 0, "tiling": False, "width": 600}
    
    if 'img' not in request.files:
        return "Missing images", 400

    image_file1 = request.files['img']

    lora_path = request.form['lora']
    print(lora_path)
    prompt = request.form['prompt']

    negative_prompt = ""
    if 'negativePrompt' in request.form:
        negative_prompt = request.form['negativePrompt']

    try:
        img_pil, img = load_image(image_file1)
    except IOError:
        return "Error: Unable to open one of the images.", 400

    input_path = "outputs/input-" + timestamp() + ".png"
    img_pil.save(input_path)
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

    # ----------------start painting--------------------------
    size = img_pil.size
    H, W = size[1], size[0]

    size = (800, 800*H//W//8*8)

    image_pil = img_pil.resize(size)
    H, W = image_pil.size[1], image_pil.size[0]
    

    payload["height"] = H
    payload["width"] = W
    payload["prompt"] = prompt + ", <lora:" + lora_path.split("/").pop().split('.')[0] + ":1>"
    print(payload["prompt"])
    payload["negative_prompt"] = negative_prompt
    payload["alwayson_scripts"]["ControlNet"]["args"][0]["image"]["image"] = encode_file_to_base64(input_path)
    lora_output_path = call_txt2img_api(**payload)

    sam_checkpoint = "sam_hq_vit_h.pth"

    device = "cuda"

    predictor = SamPredictor(build_sam_hq(checkpoint=sam_checkpoint).to(device))

    image = np.array(img_pil)

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

    
    maskname = "combined-"+timestamp()+".jpg"

    mask_pil.save("outputs/masks/maskWithPoints.jpg")
    combined = apply_mask(img_pil, mask_pil)
    combined = combined.convert("RGB")
    combined.save("outputs/masks/"+maskname)

    mask_pil = mask_pil.resize(size)
    mask_pil = mask_pil.convert("L")

    filename = "output-" + timestamp()
    for i in range(3):
        path = lora_output_path[:-5] + str(i) + lora_output_path[-4:]
        temp = Image.open(path).convert("RGB")
        temp = Image.composite(temp, image_pil, mask_pil)
        temp.save("./outputs/" + filename + "-" + str(i) + ".png")

    # result = Image.open(lora_output_path).convert("RGB")
    
    # result = Image.composite(result, image_pil, mask_pil)

    # filename = "output-" + timestamp() + ".png"
    # result.save("./outputs/" + filename)

    return {"filename": filename}, 200

@app.route('/generate-diffusers', methods=['POST'])
def lora_sd():
    # Check if both images are received
    if 'img' not in request.files:
        return "Missing images", 400

    image_file1 = request.files['img']

    depth_estimator = pipeline('depth-estimation')

    lora_path = request.form['lora']
    print(lora_path)
    prompt = request.form['prompt']

    negative_prompt = ""
    if 'negativePrompt' in request.form:
        negative_prompt = request.form['negativePrompt']

    try:
        img_pil, img = load_image(image_file1)
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

    device = "cuda"

    predictor = SamPredictor(build_sam_hq(checkpoint=sam_checkpoint).to(device))

    image = np.array(img_pil)

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


    depth_image = depth_estimator(image_pil)['depth']
    depth_image = np.array(depth_image)
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    depth_image = Image.fromarray(depth_image)
    H, W = depth_image.size[1], depth_image.size[0]
    size = (600, 600*H//W//8*8)

    depth_image = depth_image.resize(size)

    depth_image.save("lora+depth-outs/depth-" + timestamp + ".png")

    # controlnet = ControlNetModel.from_pretrained(
    #     "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16, device=device
    # ).to(device)
    print(torch.cuda.is_available())

    controlnet = ControlNetModel.from_single_file(
        "/home/jupyter/stable-diffusion-webui/extensions/sd-webui-controlnet/models/control_sd15_depth.pth", torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V2.0", controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)

    # pipe = StableDiffusionControlNetPipeline.from_single_file(
    #     "/home/jupyter/stable-diffusion-webui/models/Stable-diffusion/realisticvision.safetensors", num_in_channels=9, controlnet=controlnet, torch_dtype=torch.float16
    # ).to(device)

    # pipe = StableDiffusionControlNetPipeline.from_single_file("/home/jupyter/stable-diffusion-webui/models/Stable-diffusion/realisticvision.safetensors", controlnet=controlnet, num_in_channels=4)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    pipe.load_lora_weights(lora_path)

    pipe.enable_model_cpu_offload()

    res = pipe(prompt, depth_image, negative_prompt=negative_prompt, guidance_scale=7, num_images_per_prompt=3, num_inference_steps=20).images

    result = res[0]
    result.save("./outputs/lora-"+timestamp+".png")
    res[1].save("./outputs/lora-"+timestamp+"-1.png")
    res[2].save("./outputs/lora-"+timestamp+"-2.png")

    # mask_path = "/home/jupyter/Grounded-Segment-Anything/assets/stone/masks/black-benchtop-mask.png"
    # mask = Image.open(mask_path).convert("RGB")
    # mask = mask.resize(size)
    # enhancer = ImageEnhance.Brightness(result)
    # result = enhancer.enhance(1.1)
    # gray1 = image_pil.convert('L')
    # gray2 = result.convert('L')
    # mean1 = np.mean(np.array(gray1))
    # mean2 = np.mean(np.array(gray2))

    # adjustment_factor = mean1 / mean2
    # print(adjustment_factor)
    # enhancer = ImageEnhance.Brightness(result)
    # result = enhancer.enhance(adjustment_factor)

    mask_pil = mask_pil.convert("L")
    result = Image.composite(result, image_pil, mask_pil)

    filename = "output-" + timestamp + ".png"
    result.save("./outputs/" + filename)

    return {"filename": filename}, 200

@app.route('/generate-archive', methods=['POST'])
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
