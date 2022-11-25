import os.path as osp
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.data.data_manager import DataManager, DatasetWrapper
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, MetricMeter, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler
from datasets.adv_label_base_dataset import AdvDatasetWrapper
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import random

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, ctx_len(77), transformer.width(512)]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.dataset = cfg.DATASET.NAME
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=self.dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, [n_ctx, ctx_dim]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOT
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # label_name, ., EOT

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.classnames = classnames
        self.prompts = prompts
        self.prompt_prefix = prompt_prefix
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.token_embedding = clip_model.token_embedding  # TODO need to verify
        self.token_embedding.requires_grad_(False)

    def forward(self, adv_cns, add_adv_cn):
        label_num = self.n_cls + 1 if add_adv_cn else self.n_cls
        bsz = len(adv_cns) if add_adv_cn else 1

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(bsz * label_num, -1, -1)  # [bsz*n_cls, n_ctx, dim]

        if add_adv_cn:  # add one adv cn after original cn
            adv_cns = [adv_cn.replace('_', ' ') for adv_cn in adv_cns]
            adv_text = [self.prompt_prefix + " " + adv_cn + "." for adv_cn in adv_cns]
            tokenized_adv_text = torch.cat(
                [clip.tokenize(t, truncate=True) for t in adv_text]).cuda()  # [bsz*n_cls, ctx_len]

            with torch.no_grad():
                embedding = self.token_embedding(tokenized_adv_text).cuda()  # [bsz*n_cls, ctx_len, dim]
                adv_prefix = embedding[:, :1, :]  # SOT
                adv_suffix = embedding[:, 1 + self.n_ctx:, :]  # label_name, ., EOT, ...

            prefix = torch.zeros((bsz * label_num, self.token_prefix.size(1), self.token_prefix.size(2))).to(self.dtype).cuda()
            suffix = torch.zeros((bsz * label_num, self.token_suffix.size(1), self.token_suffix.size(2))).to(self.dtype).cuda()
            tokenized_prompts = torch.zeros((bsz * label_num, self.tokenized_prompts.size(1))).to(self.tokenized_prompts.dtype).cuda()

            for i in range(bsz):
                prefix[i * label_num: i * label_num + self.n_cls] = self.token_prefix
                prefix[i * label_num + self.n_cls] = adv_prefix[i]

                suffix[i * label_num: i * label_num + self.n_cls] = self.token_suffix
                suffix[i * label_num + self.n_cls] = adv_suffix[i]

                tokenized_prompts[i * label_num: i * label_num + self.n_cls] = self.tokenized_prompts
                tokenized_prompts[i * label_num + self.n_cls] = tokenized_adv_text[i]
        else:
            prefix = torch.cat([self.token_prefix for _ in range(bsz)], dim=0)
            suffix = torch.cat([self.token_suffix for _ in range(bsz)], dim=0)
            tokenized_prompts = torch.cat([self.tokenized_prompts for _ in range(bsz)], dim=0)

        if self.class_token_position == 'end':
            prompts = torch.cat([
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix  # (n_cls, *, dim)
            ], dim=1)

        elif self.class_token_position == 'middle':
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(label_num):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(label_num):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, tokenized_prompts

    def reset_classnames(self, classnames, clip_model):
        n_cls = len(classnames)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOT
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # label_name, ., EOT

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.classnames = classnames
        self.prompts = prompts
        self.name_lens = name_lens


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, adv_cns, add_adv_cn=False):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts, tokenized_prompts = self.prompt_learner(adv_cns, add_adv_cn)
        # prompts: [bsz*subsample_n, ctx_len(77), dim]
        # tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if add_adv_cn:
            text_features = text_features.view(
                (len(adv_cns), -1, text_features.size(-1)))  # [bsz*n_cls, dim] -> [bsz, n_cls, dim]

        logit_scale = self.logit_scale.exp()
        if add_adv_cn:
            logits = logit_scale * torch.einsum('bd,bcd->bc', image_features, text_features)
        else:
            logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        if self.cfg.ADV_LABEL_DIR and self.cfg.ADV_VOCAB_FILE:
            dataset_wrapper = AdvDatasetWrapper
        else:
            dataset_wrapper = DatasetWrapper
        self.dm = DataManager(self.cfg, dataset_wrapper=dataset_wrapper)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, adv_cn = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":  # TODO need verify
            with autocast():
                output = self.model(image, adv_cn, add_adv_cn=(self.cfg.ADV_LABEL_DIR and self.cfg.ADV_VOCAB_FILE))
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, adv_cn, add_adv_cn=(self.cfg.ADV_LABEL_DIR and self.cfg.ADV_VOCAB_FILE))
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

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
            image, label, adv_cn = self.parse_batch_test(batch)
            output = self.model(image, adv_cn, add_adv_cn=self.cfg.ADD_TEST_ADV_CN)
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
            image, label, adv_cn = self.parse_batch_test(batch)

            if reassigned_adv_cn is not None:  # re-assign a new adv classname
                adv_cn = [reassigned_adv_cn for _ in range(label.size(0))]

            output = self.model(image, adv_cn, add_adv_cn=self.cfg.ADD_TEST_ADV_CN)
            self.evaluator.process(output, label)

            pred = output.max(1)[1]
            matches = pred.eq(label).int()
            matches_index.append(matches)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

        matches_index = torch.cat(matches_index).cpu().detach().numpy().tolist()
        wrong_instance = [self.dm.dataset.test[i] for i in range(len(matches_index)) if matches_index[i] == 0]
        wrong_log = [[datum.impath, datum.classname, reassigned_adv_cn] for datum in wrong_instance]

        return list(results.values())[0], wrong_log

    # def model_inference(self, image, condition, label):
    #     return self.model(image, condition, label)

    def parse_batch_train(self, batch):
        input = batch['img']  # [bsz, channel, width, height]
        label = batch['label']  # bsz
        adv_cn = batch['adv_cn'] if 'adv_cn' in batch.keys() else None  # bsz

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, adv_cn

    def parse_batch_test(self, batch):
        input = batch['img']  # [bsz, channel, width, height]
        label = batch['label']  # bsz
        adv_cn = batch['adv_cn'] if 'adv_cn' in batch.keys() else None  # bsz

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, adv_cn

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
                                       self.epoch + 1
                               ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        if do_test and self.cfg.TEST.FINAL_MODEL == 'best_val':
            curr_result = self.test(split='val')
            if type(curr_result) is tuple:
                curr_result = curr_result[0]
            if type(self.best_result) is tuple:
                self.best_result = self.best_result[0]
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name='model-best.pth.tar'
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def after_train(self):
        print('Finished training')

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == 'best_val':
                print('Deploy the model with the best val performance')
                self.load_model(self.output_dir)
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        # Close writer
        self.close_writer()

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                                           self.max_epoch - (self.epoch + 1)
                                   ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()
