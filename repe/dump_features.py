import torch
import clip
from clip.clip import _MODELS, _download, build_model, available_models
import warnings
from typing import Union
from PIL import Image
import os
import matplotlib as mpl
from tqdm import tqdm
from collections import defaultdict
from declip import declip
import json
from collections import OrderedDict
import re
import slip.models as slip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _transform(n_px):
    return Compose([
        Resize([n_px, n_px], interpolation=BICUBIC),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


@torch.no_grad()
def dump_image_features(img_dir, classnames, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == 'clip_vit_b32':
        model, preprocess = clip.load("ViT-B/32", device=device)
    elif model_name == 'clip_vit_b16':
        model, preprocess = clip.load("ViT-B/16", device=device)
    elif model_name == 'declip_vit_b32':
        model, preprocess = declip.load("checkpoints/declip/vitb32.pth.tar", device=device)
    elif model_name == 'slip_vit_b16':
        model = slip.load('ViT-B/16/ep100', device=device)
        preprocess = _transform(model.visual.default_cfg['input_size'][-1])
    elif model_name == 'clip_rn101':
        model, preprocess = clip.load("RN101", device=device)
    else:
        raise ValueError('wrong model name!')
    folders = sorted(f.name for f in os.scandir(img_dir) if f.is_dir())

    image_features = {}
    images = defaultdict(list)
    for label, folder in enumerate(folders):
        imnames = listdir_nohidden(os.path.join(img_dir, folder))
        for imname in imnames:
            image = os.path.join(img_dir, folder, imname)
            images[folder].append(image)
    bsz = 200
    for classname, _images in tqdm(images.items()):
        _image_features = []
        for i in range(0, len(_images), bsz):
            batch = _images[i:i + bsz]
            batch = [Image.open(i) for i in batch]
            batch = [preprocess(i).unsqueeze(0).to(device) for i in batch]
            batch = torch.cat(batch)
            _image_features.append(model.encode_image(batch))
        image_features[classname] = torch.cat(_image_features)
    feature_size = sum([f.size(0) for f in image_features.values()])
    print('feature size: ', feature_size)

    torch.save(image_features, os.path.join(img_dir, f'{model_name}_image_features_grouped.pt'))


@torch.no_grad()
def dump_pretrained_text_features(dataset_dir, classnames, model_name, mode='lazy'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == 'clip_vit_b32':
        model, _ = clip.load("ViT-B/32", device=device)
    elif model_name == 'declip_vit_b32':
        model, _ = declip.load("checkpoints/declip/vitb32.pth.tar", device=device)
    elif model_name == 'slip_vit_b16_ep100':
        model = slip.load('ViT-B/16/ep100', device=device)
        # _ = _transform(model.visual.default_cfg['input_size'][-1])
    elif model_name == 'clip_rn101':
        model, _ = clip.load("RN101", device=device)
    else:
        raise ValueError('wrong model name!')

    text_features = {}
    texts = defaultdict(list)
    if mode == 'lazy':
        retrieval_dir = os.path.join(dataset_dir, 'laion5B_retrieval_1000')
        file_names = os.listdir(retrieval_dir)
        for file_name in file_names:
            classname = file_name.split('.json')[0]
            cns = classname.replace(' ', '_').split('_')
            with open(os.path.join(retrieval_dir, file_name), 'r') as f:
                retrieval_res = json.load(f)
            for res in retrieval_res:
                text = res['caption']
                for cn in cns:
                    if cn.lower() in text.lower():
                        texts[classname].append(text)
                        break
    else:
        folders = sorted(f.name for f in os.scandir(dataset_dir) if f.is_dir())
        for folder in folders:
            classname = folder.split('.parquet_outputs')[0]
            if classname in ['baluster or handrail', 'cardboard box or carton', 'product packet or packaging',
                             'shoji screen or room divider', 'swim trunks or shorts']:  # for imagenet
                classname = classname.replace(' or ', ' / ')
            if classname == 'projectile':
                classname = 'missile'
            file_names = listdir_nohidden(os.path.join(dataset_dir, folder, '00000'))

            for file_name in file_names:
                if file_name.endswith('txt'):
                    with open(os.path.join(dataset_dir, folder, '00000', file_name), 'r') as f:
                        text = f.readlines()[0]
                    cns = classname.replace(' ', '_').split('_')
                    for cn in cns:
                        if cn.lower() in text.lower():
                            texts[classname].append(text)
                            break
                    # texts[classname].append(text)
            if classname not in texts:
                print('can not retrieve', classname, 'after filter')
                texts[classname].append('a photo of a %s' % classname)
        texts['cardigan'].append('a photo of a cardigan')  # TODO
    bsz = 200
    for classname, _texts in tqdm(texts.items()):
        _text_features = []
        for i in range(0, len(_texts), bsz):
            batch = _texts[i:i + bsz]
            if 'declip' not in model_name:
                batch = torch.cat([clip.tokenize(t, truncate=True) for t in batch]).to(device)
            _text_features.append(model.encode_text(batch))
        text_features[classname] = torch.cat(_text_features)
    feature_size = sum([f.size(0) for f in text_features.values()])
    print('feature size: ', feature_size)

    torch.save(text_features, os.path.join(dataset_dir, f'{model_name}_pretrained_text_features_grouped_filtered.pt'))
