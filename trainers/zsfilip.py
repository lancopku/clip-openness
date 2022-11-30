import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from declip.filip import filip_res50, filip_vitb32
from clip.model import convert_weights
from collections import OrderedDict
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, CIFAR10_TEMPLATES, CIFAR100_TEMPLATES, \
    StanfordCars_TEMPLATES, Caltech101_TEMPLATES, DescribableTextures_TEMPLATES, EuroSAT_TEMPLATES, \
    Flowers102_TEMPLATES, Food101_TEMPLATES, SUN397_TEMPLATES, OxfordPets_TEMPLATES, UCF101_TEMPLATES, CUSTOM_TEMPLATES
import torch.distributed as dist

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "MYCIFAR10": "a photo of a {}.",
    "MYCIFAR100": "a photo of a {}.",
}

_MODELS = {
    "ViT-B/32/FILIP": "/home/renshuhuai/.cache/filip/vitb32_filip.pth.tar",
    "RN50/FILIP": "/home/renshuhuai/.cache/filip/r50_filip.pth.tar"  # TODO not yet release
}


@TRAINER_REGISTRY.register()
class ZeroshotFILIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        filip_model = self.load_filip_to_cpu(cfg)
        filip_model.to(self.device)

        if cfg.DATASET.NAME in ['FLOWERS_PETS_CARS', 'CIFAR100_CALTECH101_SUN397', 'CIFAR10_CIFAR100_ImageNet']:
            prompts = []
            for c in classnames:
                dataset_name = self.dm.dataset.class2superclass[c]
                temp = CUSTOM_TEMPLATES[dataset_name]
                prompts.append(temp.format(c.replace("_", " ")))
        else:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")

        with torch.no_grad():
            text_features = filip_model.encode_text_dense(prompts)  # [label_num, token_num, feat_dim]
            text_features /= text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.filip_model = filip_model

    def get_logits(self, dense_feat_1, selected_feat_2):
        i, j, k = dense_feat_1.shape  # [bsz, patch_num, feat_dim]
        l, m, k = selected_feat_2.shape  # [label_num, token_num, feat_dim]
        dense_feat_1 = dense_feat_1.reshape(-1, k)
        selected_feat_2 = selected_feat_2.reshape(-1, k)
        final_logits_1 = dense_feat_1 @ selected_feat_2.t()
        final_logits_1 = final_logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3)  # [bsz, label_num, patch_num. token_num]

        return final_logits_1.max(dim=-1)[0].mean(dim=-1)  # [bsz, label_num]

    def model_inference(self, image):
        image_features = self.filip_model.encode_image_dense(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = self.get_logits(image_features, self.text_features)

        # logit_scale = self.filip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def load_filip_to_cpu(self, cfg):
        backbone_name = cfg.MODEL.BACKBONE.NAME
        model_path = _MODELS[backbone_name]

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = self.build_filip_model(state_dict or model.state_dict())

        return model

    def build_filip_model(self, state_dict: dict):  # TODO ugly, should be loaded from cfg
        ckpt = state_dict
        state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            state_dict[k.replace('module.', '')] = v

        vit = "visual.proj" in state_dict
        if vit:
            image_encode = {'embed_dim': state_dict['visual.proj'].size(1)}
            text_encode = {'bpe_path': 'checkpoints/declip/bpe_simple_vocab_16e6.txt.gz',
                           'text_encode_type': 'Transformer',
                           'text_model_utils': {'random': False, 'freeze': False},
                           'embed_dim': state_dict['encode_text.text_projection.weight'].size(0)}
            clip = {'mask_rate': 0.5, 'patch_number': 14, 'use_allgather': True, 'text_mask_type': 'MLM',
                    'return_nn_bank': False, 'return_dense': True,
                    'feature_dim': state_dict['visual.proj'].size(1), 'select_topk': True}  # TODO
            kwargs = {'image_encode': image_encode, 'text_encode': text_encode, 'clip': clip}
            model = filip_vitb32(**kwargs)
        else:
            pass

        model.load_state_dict(state_dict, strict=False)

        state_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        missing_keys = model_keys - state_keys
        for k in missing_keys:
            print(f'missing key: {k}')
        return model.eval()

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == 'val' and self.val_loader is not None:
            data_loader = self.val_loader
            print('Do evaluation on {} set'.format(split))
        else:
            data_loader = self.test_loader
            print('Do evaluation on test set')

        matches_index = []
        preds = []
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

            pred = output.max(1)[1]
            matches = pred.eq(label).int()
            matches_index.append(matches)
            preds.extend(pred.cpu().detach().numpy().tolist())

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
                self.write_scalar(tag, v, self.epoch)

        matches_index = torch.cat(matches_index).cpu().detach().numpy().tolist()
        wrong_instance = [self.dm.dataset.test[i] for i in range(len(matches_index)) if matches_index[i] == 0]
        wrong_preds = [preds[i] for i in range(len(matches_index)) if matches_index[i] == 0]
        # [impath, true_classname, wrong_classname]
        wrong_log = [[datum.impath, datum.classname, self.dm.dataset.classnames[wrong_pred]] for datum, wrong_pred in
                     zip(wrong_instance, wrong_preds)]

        return list(results.values())[0], wrong_log


@TRAINER_REGISTRY.register()
class ZeroshotFILIP2(ZeroshotFILIP):
    """Prompt ensembling."""
    template_map = {'ImageNet': IMAGENET_TEMPLATES_SELECT, 'MYCIFAR10': CIFAR10_TEMPLATES,
                    'MYCIFAR100': CIFAR100_TEMPLATES, 'StanfordCars': StanfordCars_TEMPLATES,
                    'Caltech101': Caltech101_TEMPLATES, 'DescribableTextures': DescribableTextures_TEMPLATES,
                    'EuroSAT': EuroSAT_TEMPLATES, 'OxfordFlowers': Flowers102_TEMPLATES,
                    'Food101': Food101_TEMPLATES, 'SUN397': SUN397_TEMPLATES, 'OxfordPets': OxfordPets_TEMPLATES,
                    'UCF101': UCF101_TEMPLATES}

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        filip_model = self.load_filip_to_cpu(cfg)
        filip_model.to(self.device)

        for params in filip_model.parameters():
            params.requires_grad_(False)

        if cfg.DATASET.NAME in ['FLOWERS_PETS_CARS', 'CIFAR100_CALTECH101_SUN397', 'CIFAR10_CIFAR100_ImageNet']:
            mean_text_features = []
            for c in classnames:
                dataset_name = self.dm.dataset.class2superclass[c]
                templates = self.template_map[dataset_name]
                prompts = [temp.format(c.replace("_", " ")) for temp in templates]
                text_features = filip_model.encode_text_dense(prompts)  # [label_num, token_num, feat_dim]
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0, keepdim=True)
                mean_text_features.append(text_features)
            mean_text_features = torch.cat(mean_text_features)
            mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
        else:
            self.templates = self.template_map[cfg.DATASET.NAME]
            # add custom-made prompt
            # if cfg.DATASET.NAME != "ImageNet":
            #     self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

            num_temp = len(self.templates)
            print(f"Prompt ensembling (n={num_temp})")

            mean_text_features = 0
            for i, temp in enumerate(self.templates):
                prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                text_features = filip_model.encode_text_dense(prompts)  # [label_num, token_num, feat_dim]
                text_features /= text_features.norm(dim=-1, keepdim=True)
                mean_text_features = mean_text_features + text_features
            mean_text_features /= num_temp
            mean_text_features /= mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.filip_model = filip_model
