import torch
import torch.nn as nn
import os
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from collections import defaultdict
from clip import clip
from clip.model import convert_weights
from repe.repe import repe
import torch.distributed as dist
from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, CIFAR10_TEMPLATES, CIFAR100_TEMPLATES, \
    StanfordCars_TEMPLATES, Caltech101_TEMPLATES, DescribableTextures_TEMPLATES, EuroSAT_TEMPLATES, \
    Flowers102_TEMPLATES, Food101_TEMPLATES, SUN397_TEMPLATES, OxfordPets_TEMPLATES, UCF101_TEMPLATES, CUSTOM_TEMPLATES


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        if cfg.DATASET.NAME in ['FLOWERS_PETS_FOODS', 'CIFAR100_CALTECH101_SUN397', 'CIFAR10_CIFAR100_ImageNet']:
            prompts = []
            for c in classnames:
                dataset_name = self.dm.dataset.class2superclass[c]
                temp = CUSTOM_TEMPLATES[dataset_name]
                prompts.append(temp.format(c.replace("_", " ")))
        else:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        # prompts = [temp.format(c.replace("_", " "), self.dm.dataset.class2superclass[c]) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            if cfg.REPE:
                dataset_dir = self.dm.dataset.dataset_dir
                text_features = repe(dataset_dir, cfg.MODEL.BACKBONE.NAME, classnames, text_features,
                                     cfg.SHIFT_LAMBDA, cfg.RETRIEVED_NUM)
        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

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

    @torch.no_grad()
    def test_with_reassigned_adv_cn(self, split=None, reassigned_adv_cn=None):
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
        for batch_idx, batch in enumerate(data_loader):
            image, label = self.parse_batch_test(batch)

            if reassigned_adv_cn is not None:  # re-assign a new adv classname
                temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
                adv_prompt = temp.format(reassigned_adv_cn.replace("_", " "))
                adv_prompt = clip.tokenize(adv_prompt).to(self.device)
                with torch.no_grad():
                    adv_text_features = self.clip_model.encode_text(adv_prompt)
                    adv_text_features /= adv_text_features.norm(dim=-1, keepdim=True)

            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ torch.cat([self.text_features, adv_text_features]).t()

            output = logits
            self.evaluator.process(output, label)

            pred = output.max(1)[1]
            matches = pred.eq(label).int()
            matches_index.append(matches)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
                self.write_scalar(tag, v, self.epoch)
                # wandb.log({tag: v})

        matches_index = torch.cat(matches_index).cpu().detach().numpy().tolist()
        wrong_instance = [self.dm.dataset.test[i] for i in range(len(matches_index)) if matches_index[i] == 0]
        wrong_log = [[datum.impath, datum.classname, reassigned_adv_cn] for datum in wrong_instance]

        return list(results.values())[0], wrong_log


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
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

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        if cfg.DATASET.NAME in ['FLOWERS_PETS_FOODS', 'CIFAR100_CALTECH101_SUN397', 'CIFAR10_CIFAR100_ImageNet']:
            mean_text_features = []
            for c in classnames:
                dataset_name = self.dm.dataset.class2superclass[c]
                templates = self.template_map[dataset_name]
                prompts = [temp.format(c.replace("_", " ")) for temp in templates]
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                text_features = clip_model.encode_text(prompts)
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
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                text_features = clip_model.encode_text(prompts)
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
        self.clip_model = clip_model
