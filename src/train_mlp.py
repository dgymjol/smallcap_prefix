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
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import AutoTokenizer, CLIPTextModel
from transformers.models.clip.modeling_clip import _expand_mask

logging.set_verbosity_error()

data_dir = 'data/images/'
features_dir = 'features_mlp/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_name = 'openai/clip-vit-base-patch32'
feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name) 
clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)


model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

annotations = json.load(open('data/dataset_coco.json'))['images']


model_f = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_data():
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'val':
            data['val'].append({'file_name': file_name, 'cocoid': item['cocoid']})
    return data

def encode_split(data, split):
    df = pd.DataFrame(data[split])

    # h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df))):
        cocoids = df['cocoid'][idx]
        file_name = df['file_name'][idx]

        image = [Image.open(data_dir + file_name).convert("RGB")]
        with torch.no_grad(): 
            # pixel_values = feature_extractor(image, return_tensors='pt').pixel_values.to(device)
            # encodings = clip_encoder(pixel_values=pixel_values).last_hidden_state.cpu().numpy() 
            # inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
            # outputs = model_f(**inputs)

            text_inputs = tokenizer(["a photo of"], padding=True, return_tensors="pt")
            text_outputs = model(**text_inputs).last_hidden_state

            ### Upper two lines == Below lines

            output_attentions = model.text_model.config.output_attentions
            output_hidden_states = (
                model.text_model.config.output_hidden_states
            )
            return_dict = model.text_model.config.use_return_dict

            input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            ########## model.text_model.embeddings : CLIPTextEmbeddings (/home/intern06/.conda/envs/smallcap/lib/python3.9/site-packages/transformers/models/clip/modeling_clip.py)
            seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

            position_ids = model.text_model.embeddings.position_ids[:, :seq_length]
            inputs_embeds = model.text_model.embeddings.token_embedding(input_ids)

            position_embeddings = model.text_model.embeddings.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings

            hidden_states = embeddings

            ########### model.text_model : CLIPTextTransformer (/home/intern06/.conda/envs/smallcap/lib/python3.9/site-packages/transformers/models/clip/modeling_clip.py)
            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = model.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )
        
            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            encoder_outputs = model.text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = model.text_model.final_layer_norm(last_hidden_state)

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = model.text_model.final_layer_norm(last_hidden_state)

            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]


            if not return_dict:
                outputs = (last_hidden_state, pooled_output) + encoder_outputs[1:]
            else:
                outputs = BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )

        print(torch.sum(outputs.last_hidden_state.reshape(-1) == text_outputs.reshape(-1)))
        # print(torch.sum(inputs.pixel_values.reshape(-1)== pixel_values.reshape(-1).cpu())) # 150528
        # print(torch.sum(outputs.vision_model_output.last_hidden_state.reshape(-1) == torch.tensor(encodings.reshape(-1))))        #38400
        
        # breakpoint()


# def encode_split(data, split):
#     df = pd.DataFrame(data[split])

#     bs = 256
#     # h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
#     for idx in tqdm(range(0, len(df), bs)):
#         cocoids = df['cocoid'][idx:idx + bs]
#         file_names = df['file_name'][idx:idx + bs]
#         breakpoint()
#         images = [Image.open(data_dir + file_name).convert("RGB") for file_name in file_names]
#         with torch.no_grad(): 
#             pixel_values = feature_extractor(images, return_tensors='pt').pixel_values.to(device)
#             encodings = clip_encoder(pixel_values=pixel_values).last_hidden_state.cpu().numpy() 
#         for cocoid, encoding in zip(cocoids, encodings):
#             h5py_file.create_dataset(str(cocoid), (50, 768), data=encoding)

data = load_data()

encode_split(data, 'train')
encode_split(data, 'val')
