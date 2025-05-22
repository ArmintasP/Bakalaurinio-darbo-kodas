import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn.functional as F
from captum.attr import visualization as viz
from torchvision.models import VGG16_Weights
import torchvision.models as models
import pickle

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize
        (             
            mean=[0.485, 0.456, 0.406],   # R, G, B means
            std=[0.229, 0.224, 0.225]     # R, G, B stds
        )
    ])


PREPROCESS_ATTACK = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  
        transforms.Normalize
        (             
            mean=[0.485, 0.456, 0.406],   # R, G, B means
            std=[0.229, 0.224, 0.225]     # R, G, B stds
        )
    ])


DEVICE = torch.device('cuda')
VGG16_CLASSES = VGG16_Weights.DEFAULT.meta['categories']


def get_vgg():
    weights = VGG16_Weights.DEFAULT
    # Load pretrained VGG16 model
    model = models.vgg16(weights=weights).to(DEVICE)
    model.eval()
    return model


def get_image(path, prepocess_func = PREPROCESS):
    img = Image.open(path).convert('RGB')  # or .convert('L') for grayscale
    transformed_img = prepocess_func(img).unsqueeze(0).to(DEVICE)
    transformed_img_np = transformed_img.cpu().numpy()
    return transformed_img, transformed_img_np


def get_prediction(model, image, topk = 1):
    preds = model(image)
    preds = F.softmax(preds, dim=1)

    predicted_label = None
    prediction_score, pred_label_idx = torch.topk(preds, topk)

    if (topk == 1):
        predicted_label = VGG16_CLASSES[pred_label_idx]

    return pred_label_idx.squeeze_(), predicted_label, prediction_score


def get_attributions(xai, image, pred_label_idx):
    return xai.attribute(image, target=pred_label_idx)


def save_result(attributions, image, save_path, method, sign, outlier_perc = 2, use_pyplot = False):
    fig, _ = viz.visualize_image_attr(
        np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
        method=method,
        sign= sign,
        cmap= "bwr",
        alpha_overlay= 0.7,
        fig_size=(7,7),
        outlier_perc=outlier_perc,
        use_pyplot = use_pyplot)

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


def save_attack_image(modified_image, filepath):
    # Convert adversarial numpy array to torch tensor
    x_adv_tensor = torch.tensor(modified_image[0])  # Remove batch dim

    to_pil = transforms.ToPILImage()
    adv_image = to_pil(x_adv_tensor)

    adv_image.save(filepath) 


def save_pickle(savepath, filePaths, timings, attributions):
    with open(savepath, 'wb') as f:
        data = \
        {
            'filePaths': filePaths,
            'timings': timings,
            'attributions': attributions
        }
        pickle.dump(data, f)
