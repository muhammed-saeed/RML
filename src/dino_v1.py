import torch
import torchvision.transforms as transforms

feature_dim = 768


def define_model(device='cuda', model_arc='dino_resnet50'):
    dino_model = torch.hub.load('facebookresearch/dino:main', model_arc).to(device)
    dino_model = dino_model.eval()
    return dino_model


def preprocess(img):
    im_transform = get_transform()
    return im_transform(img)


def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def get_image_features(model, img_tensor):
    return model(img_tensor)
