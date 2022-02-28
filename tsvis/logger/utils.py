# -*- coding: UTF-8 -*-
"""
 Copyright 2021 Tianshu AI Platform. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =============================================================
"""
import io
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F
import logging
import functools
def check_image(tensor):
    ndim = tensor.ndim
    if ndim == 2:
        pass
    elif ndim == 3:
        if tensor.shape[2]>4:
            raise Exception(f'the expected image type is (LA), (RGB), (RGBA), and the third dimension is less than 4, '
                            f'but get shape {tensor.shape}')
    else:
        raise Exception(f'the shape of image must be (H,W) or (H,W,C), but get shape {tensor.shape}' )

def make_image(tensor):
    # Do not assume that user passes in values in [0, 255], use data type to detect
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255.0).astype(np.uint8)

    image = Image.fromarray(tensor.squeeze())

    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    return image_string

def make_histogram(values, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
    v = 1E-12
    buckets = []
    neg_buckets = []
    while v < 1E20:
        buckets.append(v)
        neg_buckets.append(-v)
        v *= 1.1
    bins = neg_buckets[::-1] + [0] + buckets

    if values.size == 0:
        raise ValueError('The input has no element.')
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(counts, pad_width=[[0, subsampling - subsampling_remainder]],
                            mode="constant", constant_values=0)
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:
    cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts

    # If start == 0, we need to add an empty one left, otherwise we can just include
    # the bin left to the first nonzero-count bin:
    counts = counts[start - 1:end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start:end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError('The histogram is empty, please file a bug report.')

    sum_sq = values.dot(values)
    return sum_sq, limits.tolist(), counts.tolist()

def make_audio(tensor, sample_rate=44100):
    import soundfile
    if abs(tensor).max() > 1:
        print('warning: audio amplitude out of range, auto clipped.')
        tensor = tensor.clip(-1, 1)
    if tensor.ndim == 1:  # old API, which expects single channel audio
        tensor = np.expand_dims(tensor, axis=1)

    assert (tensor.ndim == 2), 'Input tensor should be 2 dimensional.'
    length_frames, num_channels = tensor.shape
    assert num_channels == 1 or num_channels == 2, 'The second dimension should be 1 or 2.'

    with io.BytesIO() as fio:
        soundfile.write(fio, tensor, samplerate=sample_rate, format='wav')
        audio_string = fio.getvalue()
    return length_frames, num_channels, audio_string

def get_embedding(model,embeddings, name, model_list):
    def feature_map_hook(module, input, output):
        embeddings.append(input[0])
    try:
        for i in model._modules.keys():
            module = model._modules[i]
            if isinstance(module, nn.MaxPool2d) or (isinstance(module, nn.Conv2d) and module.stride > (1, 1)):
                module.register_forward_hook(feature_map_hook)
                for j in model_list:
                    if j[list(j.keys())[0]] == module:
                        name.append(list(j.keys())[0])
            get_embedding(module, embeddings, name, model_list)
        # for i, module in enumerate(model.children()):
        #     if list(module.children()):
        #         get_embedding(module, embeddings)
        #     elif isinstance(module, nn.MaxPool2d) or (isinstance(module, nn.Conv2d) and module.stride > (1,1)):
        #         module.register_forward_hook(feature_map_hook)

    except:
        logging.error('请下载pytorch')

def get_activation(model, input_batch, name, model_list):
    vis = None
    all_vis = []
    fmap_block = []
    grad_block = []
    model.zero_grad()
    input_batch.requires_grad_()
    def forward_hook(module, input, output):
        # activation = output
        fmap_block.append(output)
    def backward_hook(module, input, output):
        # activation_grad = output[0]
        grad_block.insert(0, output[0])
    torch.set_grad_enabled(True)
    layers_hook(model, name, model_list, forward_hook, backward_hook)
    output = model(input_batch)
    classes = torch.sigmoid(output)
    one_hot, _ = classes.max(dim=-1)
    one_hot.requires_grad_()
    model.zero_grad()
    one_hot.backward()
    for activation, activation_grad in zip(fmap_block, grad_block):
        vis = GradCam(activation, activation_grad, input_batch)
        all_vis.append(vis)
    return all_vis



def pca_decomposition(x, n_components=3):
    feats = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    feats = feats - torch.mean(feats, 0)
    u, s, v = torch.svd(feats, compute_uv=True)
    pc = torch.matmul(u[:, :n_components], torch.diag(s[:n_components]))
    pc = pc.view(x.shape[0], x.shape[2], x.shape[3], 3).permute(0, 3, 1, 2)
    return pc
def normalize_and_scale_features(features, n_sigma=1):
    scaled_features = (features - np.mean(features)) / (np.std(features))
    scaled_features = np.clip(scaled_features, -n_sigma, n_sigma)
    scaled_features = (scaled_features - scaled_features.min()) / (scaled_features.max() - scaled_features.min())
    return scaled_features

def pfv(embeddings, image_shape=None, idx_layer=None, interp_mode='bilinear'):
    if image_shape is None: image_shape = embeddings[0].shape[-2:]
    if idx_layer is None: idx_layer = len(embeddings) - 1
    with torch.no_grad():
        layer_to_visualize = pca_decomposition(embeddings[idx_layer], 3)
        amap = [F.interpolate(torch.sum(x, dim=1).unsqueeze(1), size=image_shape, mode=interp_mode) for x in embeddings[:idx_layer]]
        amap = torch.cat(amap, dim=1)
        layer_to_visualize = F.interpolate(layer_to_visualize, size=image_shape, mode=interp_mode) * torch.sum(amap,dim=1).unsqueeze(1)
        layer_to_visualize = layer_to_visualize.detach().cpu().numpy()
        rgb = normalize_and_scale_features(layer_to_visualize)
        return rgb

def layers_hook(model, name, model_list,forward_hook, backward_hook):
    for i, module in enumerate(model.children()):
        if list(module.children()) and isinstance(module, nn.Module):
            layers_hook(module, name, model_list, forward_hook, backward_hook)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            for j in model_list:
                if j[list(j.keys())[0]] == module:
                    name.append(list(j.keys())[0])

def GradCam(data_, data_grad_,img_data):
    image_size = (img_data.shape[-1], img_data.shape[-2])
    for i in range(img_data.shape[0]):
        img = img_data[i].detach().numpy()
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)
        data = data_[i, :, :, :]
        data_grad = data_grad_[i, :, :, :]  # !维度扩充
        weight = data_grad.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        mask = F.relu((weight * data).sum(dim=0))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), image_size, mode='bilinear').squeeze(0).squeeze(0)
        mask = mask.detach().numpy()
        if np.max(mask) != 0:
            mask = mask / np.max(mask)
        else:
            mask = mask
        heat_map = np.float32(map_JET(255 * mask))      #自定义映射规则
        # heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))   #opencv映射规则
        cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
    return cam

def map_JET(img):
    img = np.around(img)
    n, m = img.shape
    out = np.zeros((n, m, 3), dtype=np.uint8)
    # out = np.expand_dims(img, 2).repeat(3, 2)

    indices = np.where((img >= 0) & (img <= 31))
    values = img[indices]
    out[indices[0], indices[1], [0] * len(indices[0])] = 128 + 4 * values

    indices = np.where(img == 32)
    out[indices[0], indices[1], [0] * len(indices[0])] = 255

    indices = np.where((img >= 33) & (img <= 95))
    values = img[indices]
    out[indices[0], indices[1], [1] * len(indices[0])] = 4 + 4 * (values-33)
    out[indices[0], indices[1], [0] * len(indices[0])] = 255

    indices = np.where(img == 96)
    out[indices[0], indices[1], [2] * len(indices[0])] = 2
    out[indices[0], indices[1], [1] * len(indices[0])] = 255
    out[indices[0], indices[1], [0] * len(indices[0])] = 254

    indices = np.where((img >= 97) & (img <= 158))
    values = img[indices]
    out[indices[0], indices[1], [2] * len(indices[0])] = 6 + 4 * (values-97)
    out[indices[0], indices[1], [1] * len(indices[0])] = 255
    out[indices[0], indices[1], [0] * len(indices[0])] = 250 - 4 * (values-97)

    indices = np.where(img == 159)
    out[indices[0], indices[1], [2] * len(indices[0])] = 254
    out[indices[0], indices[1], [1] * len(indices[0])] = 255
    out[indices[0], indices[1], [0] * len(indices[0])] = 1

    indices = np.where((img >= 160) & (img <= 223))
    values = img[indices]
    out[indices[0], indices[1], [2] * len(indices[0])] = 255
    out[indices[0], indices[1], [1] * len(indices[0])] = 252 - 4 * (values-160)

    indices = np.where((img >= 224) & (img <= 255))
    values = img[indices]
    out[indices[0], indices[1], [2] * len(indices[0])] = 252 - 4 * (values-224)

    return out

def get_name_test(model):
    model_name = model.__class__.__name__
    model_list = []
    def find_parent_name(model):
        nonlocal model_name
        for i in model._modules.keys():
            module = model._modules[i]

            model_name = model_name + 'to' + module.__class__.__name__+'['+str(i)+']'
            model_list.append({model_name: module})
            find_parent_name(module)
            index = model_name.rfind("to")
            model_name = model_name[:index]
    find_parent_name(model)
    return model_list

def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            return func(*args, **kw, name=text)

        return wrapper

    return decorator

def get_name_test(model):
    model_name = model.__class__.__name__
    model_list = []
    def find_parent_name(model):
        nonlocal model_name
        for i in model._modules.keys():
            module = model._modules[i]

            model_name = model_name + 'to' + module.__class__.__name__+'['+str(i)+']'
            model_list.append({model_name: module})
            find_parent_name(module)
            index = model_name.rfind("to")
            model_name = model_name[:index]
    find_parent_name(model)
    return model_list


def all_layers(model, layers, model_list):
    for i in model._modules.keys():
        module = model._modules[i]
        if list(module.children()) and isinstance(module, nn.Module):
            all_layers(module, layers, model_list)
        else:
            layers['all_layers'].append(module)
            for j in model_list:
                if j[list(j.keys())[0]] == module:
                    layers['all_layers_name'].append(list(j.keys())[0])




