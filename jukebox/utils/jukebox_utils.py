from functools import partial
import functools
from collections import OrderedDict


import torch

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def get_forward_calls_enc_block(block, prefix=""):
    if prefix == "":
        pass
    else:
        prefix = prefix + "-"
    
    calls = []
    layer_names = []
    
    for i_block in range(len(block.model)):
        calls.append(block.model[i_block])
        layer_names.append(prefix + f"model-{i_block}")
    
    
    return calls, layer_names

def get_forward_calls_encoder(encoder, prefix=""):
    if prefix == "":
        pass
    else:
        prefix = prefix + "-"
        
    calls = []
    layer_names = []
    
    iterator = zip(list(range(encoder.levels)), encoder.downs_t, encoder.strides_t)
    for level, down_t, stride_t in iterator:
        #calls.append(encoder.level_blocks[level])
        #layer_names.append(prefix + f"level_blocks-{level}")
        
        calls_, layer_names_ = get_forward_calls_enc_block(encoder.level_blocks[level], prefix + f"level_blocks-{level}")
        
        calls += calls_
        layer_names += layer_names_
    
    
    return calls, layer_names

def get_forward_calls_decoder(decoder, prefix=""):
    if prefix == "":
        pass
    else:
        prefix = prefix + "-"
        
    calls = []
    layer_names = []
    
    iterator = reversed(list(zip(list(range(decoder.levels)), decoder.downs_t, decoder.strides_t)))
    for level, down_t, stride_t in iterator:
        #calls.append(encoder.level_blocks[level])
        #layer_names.append(prefix + f"level_blocks-{level}")
        
        calls_, layer_names_ = get_forward_calls_enc_block(decoder.level_blocks[level], prefix + f"level_blocks-{level}")
        
        calls += calls_
        layer_names += layer_names_
        
    calls.append(decoder.out)
    layer_names.append(prefix + "out")
    
    
    return calls, layer_names





def compose2func(f, g):
    return lambda *a, **kw: g(f(*a, **kw))

def compose_funclist(fs):
    if len(fs) == 0:
        return lambda x: x
    else:
        return functools.reduce(compose2func, fs)
    
                                     
def split_model(model, split_layer_name, get_calls, sequential=True):
    calls, layer_names = get_calls(model)
    assert split_layer_name in layer_names
    i = layer_names.index(split_layer_name)
    
    pre_modules = calls[:i + 1]
    pre_module_names = layer_names[:i + 1]
    post_modules = calls[i + 1:]
    post_module_names = layer_names[i + 1:]
    
    if sequential:
        pre = torch.nn.Sequential(OrderedDict([(n, l) for n, l in zip(pre_module_names, pre_modules)]))
        post = torch.nn.Sequential(OrderedDict([(n, l) for n, l in zip(post_module_names, post_modules)]))
    else:
        pre = compose_funclist(pre)
        post = compose_funclist(post)
    
    return pre, post


import torchvision
from PIL import Image


def preprocess_vqgan_input(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = torchvision.transforms.functional.resize(img, s, interpolation=Image.LANCZOS)
    img = torchvision.transforms.functional.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(torchvision.transforms.ToTensor()(img), 0)
    return 2.0 * img - 1 # Normalize to range [-1, 1]
