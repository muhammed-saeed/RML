import torch
import PIL.Image as Image
import requests

# from transformers import CLIPProcessor, CLIPModel

# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

feature_dim = 512
# def define_model(device='cuda'):
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#     return model

import clip


def define_model(device='cuda'):
    model, preprocess = clip.load("ViT-B/16", device=device)
    model = model.eval()
    return model, preprocess


# def get_image_features(model, img_tensor):
#     inputs = processor(images=img_tensor, return_tensors="pt")
#     image_features = model.get_image_features(**inputs)
#     return image_features

def get_image_features(model, img_tensor):
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
    return image_features.float()

    
    


if __name__ == '__main__':
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image.size)
    features = get_image_features(image)

    print(features.shape)
