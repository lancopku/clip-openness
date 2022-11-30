import torch
import torch.nn as nn
import torch.distributed as dist
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from slip import SLIP_VITB16, SLIP_VITL16, SLIP_VITS16
from slip.tokenizer import SimpleTokenizer
from collections import OrderedDict
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, CIFAR10_TEMPLATES, CIFAR100_TEMPLATES, \
    StanfordCars_TEMPLATES, Caltech101_TEMPLATES, DescribableTextures_TEMPLATES, EuroSAT_TEMPLATES, \
    Flowers102_TEMPLATES, Food101_TEMPLATES, SUN397_TEMPLATES, OxfordPets_TEMPLATES, UCF101_TEMPLATES, CUSTOM_TEMPLATES
from repe.repe import repe

_MODELS = {
    "ViT-B/16/ep25": "checkpoints/slip/slip_base_25ep.pt",
    "ViT-B/16/ep50": "checkpoints/slip/slip_base_50ep.pt",
    "ViT-B/16/ep100": "checkpoints/slip/slip_base_100ep.pt",
}


@TRAINER_REGISTRY.register()
class ZeroshotSLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading SLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        slip_model = self.load_slip_to_cpu(cfg)
        slip_model.to(self.device)

        tokenizer = SimpleTokenizer()

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
        prompts = tokenizer(prompts).to(self.device)

        with torch.no_grad():
            text_features = slip_model.encode_text(prompts)
            text_features = text_features / (
                        text_features.norm(dim=-1, keepdim=True) + 1e-10)  # adapted to declip source code

        self.text_features = text_features
        self.slip_model = slip_model

    def model_inference(self, image):
        image_features = self.slip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.slip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def load_slip_to_cpu(self, cfg):
        backbone_name = cfg.MODEL.BACKBONE.NAME
        model_path = _MODELS[backbone_name]

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = self.build_slip_model(state_dict or model.state_dict())

        return model

    def build_slip_model(self, state_dict: dict):
        old_args = state_dict['args']
        ckpt = state_dict
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        if old_args.model == 'SLIP_VITB16':
            model = SLIP_VITB16(rand_embed=False, ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        elif old_args.model == 'SLIP_VITS16':
            model = SLIP_VITS16(rand_embed=False, ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        elif old_args.model == 'SLIP_VITL16':
            model = SLIP_VITL16(rand_embed=False, ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        else:
            raise ValueError('not support slip model')

        model.load_state_dict(state_dict, strict=True)

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
class ZeroshotSLIP2(ZeroshotSLIP):
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

        print(f"Loading SLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        slip_model = self.load_slip_to_cpu(cfg)
        slip_model.to(self.device)

        tokenizer = SimpleTokenizer()

        for params in slip_model.parameters():
            params.requires_grad_(False)

        if cfg.DATASET.NAME in ['FLOWERS_PETS_CARS', 'CIFAR100_CALTECH101_SUN397', 'CIFAR10_CIFAR100_ImageNet']:
            mean_text_features = []
            for c in classnames:
                dataset_name = self.dm.dataset.class2superclass[c]
                templates = self.template_map[dataset_name]
                prompts = [temp.format(c.replace("_", " ")) for temp in templates]
                prompts = tokenizer(prompts).to(self.device)
                text_features = slip_model.encode_text(prompts)
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
                prompts = tokenizer(prompts).to(self.device)
                text_features = slip_model.encode_text(prompts)
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
        self.slip_model = slip_model
