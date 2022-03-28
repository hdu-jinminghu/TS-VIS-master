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
from collections import defaultdict
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


def get_gray(model, input_batch, name, model_list):
    out = []
    def forward_hook(module, input, output):
        output = output.detach().numpy()[:, 0, :, :]
        out.append(nol(output))
    def backward_hook(module, input, output):
        pass
    layers_hook(model, name, model_list, forward_hook, backward_hook)
    model(input_batch)
    return out


class Guided_backprop():
    def __init__(self, model, model_list, name, img_tensor):
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        self.model_list = model_list
        self.name = name
        self.totall_forward_data = []
        self.totall_backward_data = []
        self.input = img_tensor
        self.clearn = []

        self.first_register_hooks()
        self.need_change_layer = None

    def first_register_hooks(self):
        def first_forward_hook_fn(module, input, output):
            self.totall_forward_data.append(output)

        def first_backward_hook_fn(module, grad_in, grad_out):
            self.totall_backward_data.insert(0, grad_out[0])

        def relu_hook(model,  name, model_list, forward_hook, backward_hook):
            for i, module in enumerate(model.children()):
                if list(module.children()) and isinstance(module, nn.Module):
                    relu_hook(module, name, model_list, forward_hook, backward_hook)
                elif isinstance(module, nn.ReLU) or isinstance(module, nn.Conv2d):
                    x = module.register_forward_hook(forward_hook)
                    y = module.register_backward_hook(backward_hook)
                    self.clearn.append(x)
                    self.clearn.append(y)
                    for j in model_list:
                        if j[list(j.keys())[0]] == module:
                            name.append(list(j.keys())[0])
        relu_hook(self.model, self.name, self.model_list, first_forward_hook_fn, first_backward_hook_fn)
        model_output = self.model(self.input)
        self.model.zero_grad()
        # pred_class = model_output.argmax().item()
        #
        # # 生成目标类 one-hot 向量，作为反向传播的起点
        # grad_target_map = torch.zeros(model_output.shape,
        #                               dtype=torch.float)
        # grad_target_map[0][pred_class] = 1

        pred_class = torch.argmax(model_output, dim=1)
        # 生成目标类 one-hot 向量，作为反向传播的起点
        grad_target_map = torch.zeros(model_output.shape,
                                      dtype=torch.float)
        img_idx = torch.arange(0, model_output.size(0))
        grad_target_map[img_idx, pred_class] = 1

        # 反向传播，之前注册的 backward hook 开始起作用
        model_output.backward(grad_target_map)
        for item in self.clearn:
            item.remove()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            # 在全局变量中保存输入图片的梯度，该梯度由第一层卷积层
            # 反向传播得到，因此该函数需绑定第一个 Conv2d Layer
            self.image_reconstruction = grad_in[0]
        def forward_hook_fn(module, input, output):
            # 在全局变量中保存 ReLU 层的前向传播输出
            # 用于将来做 guided backpropagation
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop()
            if module == self.need_change_layer[list(self.need_change_layer.keys())[0]]:
                index = self.name.index(list(self.need_change_layer.keys())[0])
                need_activation = self.totall_forward_data[index]
                need_grad = self.totall_backward_data[index]
                need_activation[need_activation > 0] = 1
                positive_grad_out = torch.clamp(need_grad, min=0.0)
                new_grad_in = positive_grad_out * need_activation
                return (new_grad_in,)
            else:
                # ReLU 正向传播的输出要么大于0，要么等于0，
                # 大于 0 的部分，梯度为1，
                # 等于0的部分，梯度还是 0
                grad[grad > 0] = 1

                # grad_out[0] 表示 feature 的梯度，只保留大于 0 的部分
                positive_grad_out = torch.clamp(grad_out[0], min=0.0)
                # 创建新的输入端梯度
                new_grad_in = positive_grad_out * grad

                # ReLU 不含 parameter，输入端梯度是一个只有一个元素的 tuple
                return (new_grad_in,)
        def relu_hook(model,forward_hook, backward_hook):
            for i, module in enumerate(model.children()):
                if list(module.children()) and isinstance(module, nn.Module):
                    relu_hook(module, forward_hook, backward_hook)
                elif isinstance(module, nn.ReLU):
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)

        relu_hook(self.model, forward_hook_fn, backward_hook_fn)

    def visualize(self, input_image):
        # 获取输出，之前注册的 forward hook 开始起作用
        model_output = self.model(input_image)
        self.model.zero_grad()
        # pred_class = model_output.argmax().item()
        #
        # # 生成目标类 one-hot 向量，作为反向传播的起点
        # grad_target_map = torch.zeros(model_output.shape,
        #                               dtype=torch.float)
        # grad_target_map[0][pred_class] = 1
        pred_class = torch.argmax(model_output, dim=1)
        # 生成目标类 one-hot 向量，作为反向传播的起点
        grad_target_map = torch.zeros(model_output.shape,
                                      dtype=torch.float)
        img_idx = torch.arange(0, model_output.size(0))
        grad_target_map[img_idx, pred_class] = 1
        # 反向传播，之前注册的 backward hook 开始起作用
        model_output.backward(grad_target_map)
        self.image_reconstruction = input_image.grad
        # 得到 target class 对输入图片的梯度，转换成图片格式
        result = self.image_reconstruction.data.permute(0, 2, 3, 1)
        return result.numpy()

    @staticmethod  ##使得normalize变成静态方法和他model没有关系，Guided_backprop.normalize()调用
    def normalize(I):
        # 归一化梯度map，先归一化到 mean=0 std=1
        norm = (I - I.mean()) / I.std()
        # 把 std 重置为 0.1，让梯度map中的数值尽可能接近 0
        norm = norm * 0.1
        # 均值加 0.5，保证大部分的梯度值为正
        norm = norm + 0.5
        # 把 0，1 以外的梯度值分别设置为 0 和 1
        norm = norm.clip(0, 1)
        return norm
    def find_layer(self):
        all_data = []
        for name in self.name:
            for j in self.model_list:
                if name == list(j.keys())[0]:
                    self.need_change_layer = j
                    self.register_hooks()
                    result = self.visualize(self.input)
                    result = self.normalize(result) * 255
                    all_data.append(result)
                    break
        return all_data













def find_output_sorce(output):
    datas = []
    # output = torch.rand(10, 12)
    sort_index = torch.argsort(output, dim=1, descending=True)

    if output.shape[1] >10:
        sort_index = sort_index[:, :10]
    sort_values = torch.gather(output, 1, sort_index)
    datas.append(sort_index.detach().numpy())
    datas.append(sort_values.detach().numpy())
    return np.array(datas)




def nol(data):
    nb, nn, nf = data.shape
    data = data.reshape(nb, -1)
    min_d, max_d = np.expand_dims(np.min(data, axis=-1), axis=1), np.expand_dims(np.max(data, axis=-1),axis=1)
    data = (data-min_d)/(max_d-min_d)*255
    return data.reshape(nb, nn, nf)

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

def get_attention(model, model_type, tokenizer, sentence_a, sentence_b=None, include_queries_and_keys=False):
    """Compute representation of attention to pass to the d3 visualization

        Args:
            model: pytorch-transformers model
            model_type: type of model. Valid values 'bert', 'gpt2', 'xlnet', 'roberta'
            tokenizer: pytorch-transformers tokenizer
            sentence_a: Sentence A string
            sentence_b: Sentence B string
            include_queries_and_keys: Indicates whether to include queries/keys in results

        Returns:
          Dictionary of attn representations with the structure:
          {
            'all': All attention (source = AB, target = AB)
            'aa': Sentence A self-attention (source = A, target = A) (if sentence_b is not None)
            'bb': Sentence B self-attention (source = B, target = B) (if sentence_b is not None)
            'ab': Sentence A -> Sentence B attention (source = A, target = B) (if sentence_b is not None)
            'ba': Sentence B -> Sentence A attention (source = B, target = A) (if sentence_b is not None)
          }
          where each value is a dictionary:
          {
            'left_text': list of source tokens, to be displayed on the left of the vis
            'right_text': list of target tokens, to be displayed on the right of the vis
            'attn': list of attention matrices, one for each layer. Each has shape [num_heads, source_seq_len, target_seq_len]
            'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
            'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
          }
        """

    if model_type not in ('bert', 'gpt2', 'xlnet', 'roberta'):
        raise ValueError("Invalid model type:", model_type)
    if not sentence_a:
        raise ValueError("Sentence A is required")
    is_sentence_pair = bool(sentence_b)
    if is_sentence_pair and model_type not in ('bert', 'roberta', 'xlnet'):
        raise ValueError(f'Model {model_type} does not support sentence pairs')
    if is_sentence_pair and model_type == 'xlnet':
        raise NotImplementedError("Sentence-pair inputs for XLNet not currently supported.")

    # Prepare inputs to model
    tokens_a = None
    tokens_b = None
    token_type_ids = None
    if not is_sentence_pair:  # Single sentence
        if model_type in ('bert', 'roberta'):
            tokens_a = [tokenizer.cls_token] + tokenizer.tokenize(sentence_a) + [tokenizer.sep_token]
        elif model_type == 'xlnet':
            tokens_a = tokenizer.tokenize(sentence_a) + [tokenizer.sep_token] + [tokenizer.cls_token]
        else:
            tokens_a = tokenizer.tokenize(sentence_a)
    else:
        if model_type == 'bert':
            tokens_a = [tokenizer.cls_token] + tokenizer.tokenize(sentence_a) + [tokenizer.sep_token]
            tokens_b = tokenizer.tokenize(sentence_b) + [tokenizer.sep_token]
            token_type_ids = torch.LongTensor([[0] * len(tokens_a) + [1] * len(tokens_b)])
        elif model_type == 'roberta':
            tokens_a = [tokenizer.cls_token] + tokenizer.tokenize(sentence_a) + [tokenizer.sep_token]
            tokens_b = [tokenizer.sep_token] + tokenizer.tokenize(sentence_b) + [tokenizer.sep_token]
            # Roberta doesn't use token type embeddings per https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/convert_roberta_checkpoint_to_pytorch.py
        else:
            tokens_b = tokenizer.tokenize(sentence_b)

    token_ids = tokenizer.convert_tokens_to_ids(tokens_a + (tokens_b if tokens_b else []))
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)

    # Call model to get attention data
    model.eval()
    if token_type_ids is not None:
        output = model(tokens_tensor, token_type_ids=token_type_ids)
    else:
        output = model(tokens_tensor)
    attn_data_list = output[-1]

    # Populate map with attn data and, optionally, query, key data
    attn_dict = defaultdict(list)
    if include_queries_and_keys:
        queries_dict = defaultdict(list)
        keys_dict = defaultdict(list)

    if is_sentence_pair:
        slice_a = slice(0, len(tokens_a))  # Positions corresponding to sentence A in input
        slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b))  # Position corresponding to sentence B in input
    for layer, attn_data in enumerate(attn_data_list):
        # Process attention
        attn = attn_data['attn'][0]  # assume batch_size=1; shape = [num_heads, source_seq_len, target_seq_len]
        attn_dict['all'].append(attn.tolist())
        if is_sentence_pair:
            attn_dict['aa'].append(
                attn[:, slice_a, slice_a].tolist())  # Append A->A attention for layer, across all heads
            attn_dict['bb'].append(
                attn[:, slice_b, slice_b].tolist())  # Append B->B attention for layer, across all heads
            attn_dict['ab'].append(
                attn[:, slice_a, slice_b].tolist())  # Append A->B attention for layer, across all heads
            attn_dict['ba'].append(
                attn[:, slice_b, slice_a].tolist())  # Append B->A attention for layer, across all heads
        # Process queries and keys
        if include_queries_and_keys:
            queries = attn_data['queries'][0]  # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
            keys = attn_data['keys'][0]  # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
            queries_dict['all'].append(queries.tolist())
            keys_dict['all'].append(keys.tolist())
            if is_sentence_pair:
                queries_dict['a'].append(queries[:, slice_a, :].tolist())
                keys_dict['a'].append(keys[:, slice_a, :].tolist())
                queries_dict['b'].append(queries[:, slice_b, :].tolist())
                keys_dict['b'].append(keys[:, slice_b, :].tolist())

    tokens_a = format_special_chars(tokens_a)
    if tokens_b:
        tokens_b = format_special_chars(tokens_b)
    if model_type != 'gpt2':
        tokens_a = format_delimiters(tokens_a, tokenizer)
        if tokens_b:
            tokens_b = format_delimiters(tokens_b, tokenizer)

    results = {
        'all': {
            'attn': attn_dict['all'],
            'left_text': tokens_a + (tokens_b if tokens_b else []),
            'right_text': tokens_a + (tokens_b if tokens_b else [])
        }
    }
    if is_sentence_pair:
        results.update({
            'aa': {
                'attn': attn_dict['aa'],
                'left_text': tokens_a,
                'right_text': tokens_a
            },
            'bb': {
                'attn': attn_dict['bb'],
                'left_text': tokens_b,
                'right_text': tokens_b
            },
            'ab': {
                'attn': attn_dict['ab'],
                'left_text': tokens_a,
                'right_text': tokens_b
            },
            'ba': {
                'attn': attn_dict['ba'],
                'left_text': tokens_b,
                'right_text': tokens_a
            }
        })
    if include_queries_and_keys:
        results['all'].update({
            'queries': queries_dict['all'],
            'keys': keys_dict['all'],
        })
        if is_sentence_pair:
            results['aa'].update({
                'queries': queries_dict['a'],
                'keys': keys_dict['a'],
            })
            results['bb'].update({
                'queries': queries_dict['b'],
                'keys': keys_dict['b'],
            })
            results['ab'].update({
                'queries': queries_dict['a'],
                'keys': keys_dict['b'],
            })
            results['ba'].update({
                'queries': queries_dict['b'],
                'keys': keys_dict['a'],
            })
    return results

def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ') for t in tokens]

def format_delimiters(tokens, tokenizer):
    formatted_tokens = []
    for t in tokens:
        if tokenizer.sep_token:
            t = t.replace(tokenizer.sep_token, '[SEP]')
        if tokenizer.cls_token:
            t = t.replace(tokenizer.cls_token, '[CLS]')
        formatted_tokens.append(t)
    return formatted_tokens








