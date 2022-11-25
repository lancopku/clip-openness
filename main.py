import argparse
import copy
import random

import torch
import io
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import sys
import json
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
from tqdm import tqdm
from dassl.evaluation import build_evaluator
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.coop import PromptLearner, load_clip_to_cpu
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.stanford_cars
import datasets.sun397
import datasets.caltech101
import datasets.imagenet
import datasets.imagenet_entity13
import datasets.imagenet_living17
import datasets.my_cifar10
import datasets.my_cifar100
import datasets.flowers_pets_cars
import datasets.cifar100_caltech101_sun397
import datasets.cifar100_caltech101_sun397
import datasets.cifar10_cifar100_imagenet

import trainers.coop
import trainers.zsclip
import trainers.zsdeclip
import trainers.zsslip
import trainers.zsfilip
import trainers.zsdefilip

_tokenizer = _Tokenizer()


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    cfg.LOCAL_RANK = args.local_rank
    cfg.ADV_LABEL_DIR = args.adv_label_dir
    cfg.ADV_VOCAB_FILE = args.adv_vocab_file
    cfg.ADD_TEST_ADV_CN = args.add_test_adv_cn
    cfg.TARGET_LABEL_DIR = None
    cfg.REPE = args.repe
    cfg.SHIFT_LAMBDA = args.shift_lambda
    cfg.RETRIEVED_NUM = args.retrieved_num
    cfg.LOG_WRONG_PRED = args.log_wrong_prediction


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # cfg.freeze()

    return cfg


def adv_vocab_mining(trainer):
    adv_class_res = {}
    with open(args.adv_vocab_file, 'r') as f:
        for line in f.readlines():
            adv_cn = line.strip()
            if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
                print('adv cn: ', adv_cn)
            acc, wrong_log = trainer.test_with_reassigned_adv_cn(reassigned_adv_cn=adv_cn)
            adv_class_res[adv_cn] = {'acc': acc, 'wrong_log': wrong_log}

    adv_class_res = sorted(adv_class_res.items(), key=lambda kv: (kv[1]['acc'], kv[0]))
    json_str = json.dumps(adv_class_res, indent=2)
    with open(f'{args.output_dir}/adv_class_res.json', 'w') as json_file:
        json_file.write(json_str)


def incremental_evaluation(cfg, trainer, clip_model):
    """
    We calculate Acc-E and Acc-S together
    """
    base_dir = args.target_label_dir
    superclass2class = trainer.dm.dataset.superclass2class
    acc_c, acc_e, acc_s = [], [], []
    for target_vocab in superclass2class.keys():  # target vocabulary
        non_target_vocabs = list(superclass2class.keys())
        non_target_vocabs.remove(target_vocab)  # take the rest as non-target vocabularies

        classnames = copy.deepcopy(superclass2class[target_vocab])
        target_vocab_dir = os.path.join(base_dir, target_vocab.replace(' ', ''))
        trainer = reset_trainer(cfg, trainer, clip_model, target_vocab, target_vocab_dir, classnames)
        # evaluate model
        acc, wrong_log, conditional_acc = trainer.test(target_vocab=superclass2class[target_vocab])
        acc_c.append(acc)

        acc_e_local, acc_s_local = [], []  # local metric for a given target vocab

        for i in tqdm(range(args.trials)):
            classnames = copy.deepcopy(superclass2class[target_vocab])
            acc_e_trial_i, acc_s_trial_i = [acc, ], []  # metric for each trial

            # we generate a permutation of non-target vocabs for each trial
            non_target_vocabs_permuted = np.random.permutation(non_target_vocabs).tolist()
            for j, vocab in enumerate(non_target_vocabs_permuted):
                vocab_dir = os.path.join(base_dir, target_vocab.replace(' ', ''), 'trial_%d' % i, str(j))
                classnames += superclass2class[vocab]  # incrementally extend vocab
                trainer = reset_trainer(cfg, trainer, clip_model, vocab, vocab_dir, classnames)
                # evaluate model
                acc, wrong_log, conditional_acc = trainer.test(target_vocab=superclass2class[target_vocab])
                acc_e_trial_i.append(acc)
                acc_s_trial_i.append(conditional_acc)
                if args.log_wrong_prediction:
                    json_str = json.dumps(wrong_log, indent=2)
                    with open(f'{vocab_dir}/wrong_log.json', 'w') as json_file:
                        json_file.write(json_str)
                    sys.stdout.file.close()
            acc_e_trial_i = sum(acc_e_trial_i) / len(acc_e_trial_i)
            acc_s_trial_i = sum(acc_s_trial_i) / len(acc_s_trial_i)
            acc_e_local.append(acc_e_trial_i)
            acc_s_local.append(acc_s_trial_i)
        acc_e_local = sum(acc_e_local) / len(acc_e_local)
        acc_s_local = sum(acc_s_local) / len(acc_s_local)
        acc_e.append(acc_e_local)
        acc_s.append(acc_s_local)

    try:
        if isinstance(sys.stdout, io.TextIOWrapper):
            setup_logger(base_dir)  # initialization logger
        sys.stdout.file = open(os.path.join(base_dir, 'log.txt'), 'a')
    except:
        pass
    for target_vocab, acc_c_local, acc_e_local, acc_s_local in zip(superclass2class.keys(), acc_c, acc_e, acc_s):
        print(f"For {target_vocab}, Acc-C (local): {acc_c_local}, Acc-E (local): {acc_e_local}, Acc-S (local): {acc_s_local}")
    acc_c = sum(acc_c) / len(acc_c)
    acc_e = sum(acc_e) / len(acc_e)
    acc_s = sum(acc_s) / len(acc_s)
    print(f"For {cfg.DATASET.NAME}, Acc-C: {acc_c}, Acc-E: {acc_e}, Acc-S: {acc_s}")


def reset_trainer(cfg, trainer, clip_model, vocab, vocab_dir, classnames):
    """
    reset trainer after vocab extension
    """
    # dump target vocab and non-target vocab into file
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    json_str = json.dumps(classnames, indent=2)
    with open(f'{vocab_dir}/target_label.json', 'w') as json_file:
        json_file.write(json_str)

    # reset trainer
    # cfg.VOCABS = vocabs[:vocabs.index(vocab) + 1]  # TODO
    cfg.TARGET_LABEL_DIR = vocab_dir
    cfg.OUTPUT_DIR = vocab_dir
    if isinstance(sys.stdout, io.TextIOWrapper):
        setup_logger(cfg.OUTPUT_DIR)  # initialization logger
    sys.stdout.file = open(os.path.join(cfg.OUTPUT_DIR, 'log.txt'), 'w')
    trainer.build_data_loader()
    if args.trainer == 'CoOp':
        clip_model.cpu()
        trainer.model.prompt_learner.reset_classnames(classnames, clip_model)
        trainer.model.to(trainer.device)
    else:
        trainer.build_model()
    trainer.evaluator = build_evaluator(cfg, lab2cname=trainer.dm.lab2cname)
    return trainer


def main(args):
    np.random.seed(42)
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        if args.trainer == 'CoOp':
            clip_model = load_clip_to_cpu(cfg)
            if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
                # CLIP's default precision is fp16
                clip_model.float()
        else:
            # token_prefix, token_suffix, ctx, prompts = None, None, None, None
            clip_model = None

        if args.reassign_test_adv_cn:
            adv_vocab_mining(trainer)

        elif args.incremental_evaluation:
            incremental_evaluation(cfg, trainer, clip_model)

        else:
            if isinstance(sys.stdout, io.TextIOWrapper):
                setup_logger(args.output_dir)  # initialization logger
            sys.stdout.file = open(os.path.join(args.output_dir, 'log.txt'), 'a')

            acc, wrong_log, conditional_acc = trainer.test()

            if args.log_wrong_prediction:
                json_str = json.dumps(wrong_log, indent=2)
                with open(f'{args.output_dir}/wrong_log.json', 'w') as json_file:
                    json_file.write(json_str)
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reassign-test-adv-cn", action="store_true")
    parser.add_argument("--incremental-evaluation", action="store_true")
    parser.add_argument("--trials", type=int, default=1, help="trials num of permutation of extension vocabulary")
    parser.add_argument("--add-test-adv-cn", action="store_true")
    parser.add_argument("--log-wrong-prediction", action="store_true")
    parser.add_argument("--adv-label-dir", type=str)
    parser.add_argument("--adv-vocab-file", type=str)
    parser.add_argument("--target-label-dir", type=str)
    parser.add_argument("--repe", action="store_true", help="retrieval-enhanced prompt engineering")
    parser.add_argument("--pretrained-feature-dir", type=str)
    parser.add_argument("--shift-lambda", type=float, default=1)
    parser.add_argument("--retrieved-num", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
