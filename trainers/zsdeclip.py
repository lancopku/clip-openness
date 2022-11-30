import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from declip.declip import declip_res50, declip_vitb32
from clip.model import convert_weights
from repe.repe import repe
from collections import OrderedDict
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, CIFAR10_TEMPLATES, CIFAR100_TEMPLATES, \
    StanfordCars_TEMPLATES, Caltech101_TEMPLATES, DescribableTextures_TEMPLATES, EuroSAT_TEMPLATES, \
    Flowers102_TEMPLATES, Food101_TEMPLATES, SUN397_TEMPLATES, OxfordPets_TEMPLATES, UCF101_TEMPLATES, CUSTOM_TEMPLATES
import torch.distributed as dist

_MODELS = {
    "ViT-B/32/DeClip": "checkpoints/declip/vitb32.pth.tar",
    "RN50/DeClip": "checkpoints/declip/r50.pth.tar"
}


@TRAINER_REGISTRY.register()
class ZeroshotDeCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading DeCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        declip_model = self.load_declip_to_cpu(cfg)
        declip_model.to(self.device)

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
            text_features = declip_model.encode_text(prompts)
            text_features = text_features / (
                        text_features.norm(dim=-1, keepdim=True) + 1e-10)  # adapted to declip source code

        self.text_features = text_features
        self.declip_model = declip_model

    def model_inference(self, image):
        image_features = self.declip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.declip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def load_declip_to_cpu(self, cfg):
        backbone_name = cfg.MODEL.BACKBONE.NAME
        model_path = _MODELS[backbone_name]

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = self.build_declip_model(state_dict or model.state_dict())

        return model

    def build_declip_model(self, state_dict: dict):  # TODO ugly, should be loaded from cfg
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
            clip = {'use_allgather': True, 'text_mask_type': 'MLM', 'return_nn_bank': True,
                    'feature_dim': state_dict['visual.proj'].size(1)}
            kwargs = {'image_encode': image_encode, 'text_encode': text_encode, 'clip': clip}
            model = declip_vitb32(**kwargs)
        else:
            image_encode = {'bn_group_size': 1, 'bn_sync_stats': True, 'embed_dim': 1024}  # TODO 32 in config?
            text_encode = {'bpe_path': 'checkpoints/declip/bpe_simple_vocab_16e6.txt.gz',
                           'text_encode_type': 'Transformer',
                           'text_model_utils': {'random': False, 'freeze': False}, 'embed_dim': 2014}
            clip = {'use_allgather': True, 'text_mask_type': 'MLM', 'return_nn_bank': True}
            kwargs = {'image_encode': image_encode, 'text_encode': text_encode, 'clip': clip}
            model = declip_res50(**kwargs)

        model.load_state_dict(state_dict, strict=False)

        state_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        missing_keys = model_keys - state_keys
        for k in missing_keys:
            print(f'missing key: {k}')
        return model.eval()

    @torch.no_grad()
    def test(self, split=None, target_vocab=None):
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

        if target_vocab is not None:
            cname2lab = {v: k for k, v in self.dm.dataset.lab2cname.items()}
            target_acc = []
            for target_class in target_vocab:
                target_label = cname2lab[target_class]
                target_res = self.evaluator._per_class_res[target_label]
                target_acc.append((100. * sum(target_res) / len(target_res)))
            conditional_acc = sum(target_acc) / len(target_acc)
        else:
            conditional_acc = 0

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
                self.write_scalar(tag, v, self.epoch)

        if self.cfg.LOG_WRONG_PRED:
            matches_index = torch.cat(matches_index).cpu().detach().numpy().tolist()
            wrong_instance = [self.dm.dataset.test[i] for i in range(len(matches_index)) if matches_index[i] == 0]
            wrong_preds = [preds[i] for i in range(len(matches_index)) if matches_index[i] == 0]
            # [impath, true_classname, wrong_classname]
            wrong_log = [[datum.impath, datum.classname, self.dm.dataset.classnames[wrong_pred]] for datum, wrong_pred
                         in zip(wrong_instance, wrong_preds)]
        else:
            wrong_log = None

        return list(results.values())[0], wrong_log, conditional_acc


@TRAINER_REGISTRY.register()
class ZeroshotDeCLIP2(ZeroshotDeCLIP):
    """Prompt ensembling."""
    template_map = {'ImageNet': IMAGENET_TEMPLATES_SELECT, 'MYCIFAR10': CIFAR10_TEMPLATES,
                    'MYCIFAR100': CIFAR100_TEMPLATES, 'StanfordCars': StanfordCars_TEMPLATES,
                    'Caltech101': Caltech101_TEMPLATES, 'DescribableTextures': DescribableTextures_TEMPLATES,
                    'EuroSAT': EuroSAT_TEMPLATES, 'OxfordFlowers': Flowers102_TEMPLATES,
                    'Food101': Food101_TEMPLATES, 'SUN397': SUN397_TEMPLATES, 'OxfordPets': OxfordPets_TEMPLATES,
                    'UCF101': UCF101_TEMPLATES, 'ImageNet_Entity13': IMAGENET_TEMPLATES_SELECT,
                    'ImageNet_Living17': IMAGENET_TEMPLATES_SELECT}

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading DeCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        declip_model = self.load_declip_to_cpu(cfg)
        declip_model.to(self.device)

        for params in declip_model.parameters():
            params.requires_grad_(False)

        if cfg.DATASET.NAME in ['FLOWERS_PETS_CARS', 'CIFAR100_CALTECH101_SUN397', 'CIFAR10_CIFAR100_ImageNet']:
            mean_text_features = []
            for c in classnames:
                dataset_name = self.dm.dataset.class2superclass[c]
                templates = self.template_map[dataset_name]
                prompts = [temp.format(c.replace("_", " ")) for temp in templates]
                text_features = declip_model.encode_text(prompts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0, keepdim=True)
                mean_text_features.append(text_features)
            mean_text_features = torch.cat(mean_text_features)
            mean_text_features /= mean_text_features.norm(dim=-1, keepdim=True)
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
                text_features = declip_model.encode_text(prompts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                if cfg.REPE:
                    dataset_dir = self.dm.dataset.dataset_dir
                    text_features = repe(dataset_dir, cfg.MODEL.BACKBONE.NAME, classnames, text_features,
                                         cfg.SHIFT_LAMBDA, cfg.RETRIEVED_NUM)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                mean_text_features = mean_text_features + text_features
            mean_text_features /= num_temp
            mean_text_features /= mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.declip_model = declip_model
