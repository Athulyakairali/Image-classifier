import argparse
import json
import PIL
import torch
import numpy as np
from torchvision import models

def argument_parser():
    parser = argparse.ArgumentParser(description="Image Classifier Udacity Project")
    parser.add_argument('--image_path', type=str, help='Path to the image for prediction.', required=True)
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file.', required=True)
    parser.add_argument('--top_k', type=int, help='Number of top classes to display.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='class_mapping.json')
    parser.add_argument('--use_gpu', default="gpu", action="store_true", dest="use_gpu")

    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    img = PIL.Image.open(image_path)

    img = img.resize((256, 256))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    img = img.crop((left, top, right, bottom))

    img = np.array(img) / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))

    return img

def predict(image, model, device, class_mapping, top_k=5):
    model.to(device)
    image = torch.FloatTensor(image).to(device)
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)

    top_ps, top_class = ps.topk(top_k, dim=1)

    top_ps = top_ps.cpu().numpy().flatten()
    top_class = top_class.cpu().numpy().flatten()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_class]
    top_classes = [class_mapping[label] for label in top_labels]

    return top_ps, top_labels, top_classes

def display_results(probs, classes):
    for i, (class_name, prob) in enumerate(zip(classes, probs)):
        print("Rank {}: Class: {}, Probability: {:.2%}".format(i + 1, class_name, prob))

def main():
    args = argument_parser()
    
    with open(args.category_names, 'r') as f:
        class_mapping = json.load(f)

    model = load_checkpoint(args.checkpoint_path)
    
    image_tensor = process_image(args.image_path)
    
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    top_probs, top_labels, top_classes = predict(image_tensor, model, device, class_mapping, args.top_k)
    
    display_results(top_probs, top_classes)

if __name__ == '__main__':
    main()
