import os
import sys
import pandas as pd
import json
from tqdm import tqdm 
from PIL import Image
import torch
from multiprocessing import Pool
import h5py
from transformers import logging
from transformers import CLIPFeatureExtractor, CLIPVisionModel
from transformers import CLIPProcessor, CLIPModel
from torch import nn
from model1.clip import _transform, load
from model1.model import convert_weights, CLIP, IM2TEXT

logging.set_verbosity_error()

data_dir = 'data/images/'
features_dir = 'i2t_features_77/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import open_clip

tokenize = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')

annotations = json.load(open('data/dataset_coco.json'))['images']

clip, _, preprocess = load('ViT-L/14', jit=False)
clip = clip.to(device)

img2text = IM2TEXT(embed_dim=768, middle_dim=512, output_dim=768)
img2text = img2text.to(device)

img2text_weight = torch.load('pic2word.pt')['state_dict_img2text']
for key in list(img2text_weight.keys()):
    new_key = key.replace("module.", "")
    img2text_weight[new_key] = img2text_weight.pop(key)
img2text.load_state_dict(img2text_weight)
print("------- LOAD IMG2TEXT PRETRAINED WEIGHT")

def load_data():
    data = {'train': [], 'val': [], 'test': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'val':
            data['val'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'test' :
            data['test'].append({'file_name': file_name, 'cocoid': item['cocoid']})

    return data

def encode_split(data, split):
    df = pd.DataFrame(data[split])

    bs = 256
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        cocoids = df['cocoid'][idx:idx + bs]
        file_names = df['file_name'][idx:idx + bs]
        images = torch.stack([preprocess(Image.open(data_dir + file_name).convert("RGB")) for file_name in file_names], dim=0).to(device)
        with torch.no_grad(): 

            image_features = clip.encode_image(images)
            token_features = img2text(image_features.float())

            text = tokenize("a photo of")
            text = text.to(device)
            text = text.view(1, -1)
            text = text.repeat(token_features.size(0), 1)

            img_tokens = token_features

            b_size = img_tokens.size(0)
            x = clip.token_embedding(text)  # [batch_size, n_ctx, d_model]
            collect_ind = text == clip.end_id 
            collect_ind = collect_ind.nonzero()[:, 1]
            img_tokens = img_tokens.view(b_size, 1, -1)
            x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-1]], dim=1)
            x = x + clip.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = clip.transformer(x.half())
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = clip.ln_final(x)
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)    
            # x = x[torch.arange(x.size(0)), collect_ind+1] @ clip.text_projection

            x = x.float().cpu().numpy()

        # for cocoid, encoding in zip(cocoids, encodings):
            # h5py_file.create_dataset(str(cocoid), (50, 768), data=encoding)
        for cocoid, encoding in zip(cocoids, x):
            h5py_file.create_dataset(str(cocoid), (77, 768), data=encoding)

data = load_data()

encode_split(data, 'train')
encode_split(data, 'val')
# encode_split(data, 'test')

