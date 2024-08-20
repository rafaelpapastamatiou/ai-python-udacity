import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json

def parse_input_args():
    parser = argparse.ArgumentParser(description="AIPND Predict app")

    parser.add_argument("image_path", type=str, help="Path of the image")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint to be used")

    parser.add_argument("--top_k", type=int, default=5, help="Number of classes to be used when returning probabilities")
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Path to the JSON dict used to map the classes to their names"
    )
    parser.add_argument("--gpu", action='store_true', default=False, help="Use GPU (Cuda) if available")
    
    return parser.parse_args()

def load_category_names(category_names_filepath):
    with open(category_names_filepath, 'r') as f:
        category_names = json.load(f)
    
    return category_names
    
def load_checkpoint(checkpoint_filepath):
    checkpoint = torch.load(checkpoint_filepath)
    
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = {str(v): k for k, v in model.class_to_idx.items()}
    
    print("Model", checkpoint['arch'], "loaded successfully")
    
    return model

def process_image(image):
    # Resize Image
    size = 256
    
    w, h = image.size
    w_ratio = w / h
    h_radio = h / w
    
    new_w = size if w <= h else int(w_ratio *  size)
    new_h = size if h < w else int(h_radio *  size)
    
    image = image.resize((new_w, new_h))
    
    # Crop Image
    crop_size = 224
    
    left = (new_w - crop_size) / 2
    right = (new_w + crop_size) / 2
    top = (new_h - crop_size) / 2
    bottom = (new_h + crop_size) / 2
        
    image = image.crop((left, top, right, bottom))

    # Normalize image
    np_image = np.array(image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image / 255 - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def pil_image_to_tensor(pil_image):
    img = process_image(pil_image)
    return torch.FloatTensor(img)
    
def predict(
    image_path,
    model,
    category_names,
    top_k=5,
    gpu = False
):
    device = torch.device("cuda" if gpu == True and torch.cuda.is_available() else "cpu")
    
    print("Predicting on", device)

    model.eval()
    model.to(device)
    
    pil_image = Image.open(image_path)
    
    img = pil_image_to_tensor(pil_image)
    img = img.to(device)
    img = torch.unsqueeze(img, 0)
    
    with torch.no_grad():
        logps = model.forward(img)
        ps = torch.exp(logps)
    
    top_ps, top_classes = ps.topk(top_k, dim=1)

    top_classes = top_classes[0]
    top_ps = top_ps[0].cpu().numpy()

    top_class_idx = [model.idx_to_class[str(i.item())] for i in top_classes]
    top_classes = [category_names[str(i)] for i in top_class_idx]
    
    return top_ps, top_classes

def main():
    args = parse_input_args()
    
    print(args)
    
    category_names = load_category_names(args.category_names)
    
    model = load_checkpoint(args.checkpoint)
    
    top_ps, top_classes = predict(
        args.image_path,
        model,
        category_names,
        top_k=args.top_k,
        gpu=args.gpu
    )
    
    for i in range(len(top_ps)):
        print("Name:", top_classes[i], "Probability:", str(round(top_ps[i].item() * 100, 2)), "%")
    
if __name__ == "__main__":
    main()
