import os
import pickle
from collections import OrderedDict
import numpy as np
import random
import json
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from .adv_label_base_dataset import AdvDatum
from dassl.utils import listdir_nohidden


@DATASET_REGISTRY.register()
class MYCIFAR10(DatasetBase):

    dataset_dir = "cifar10"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.adv = True if cfg.ADV_LABEL_DIR and cfg.ADV_VOCAB_FILE else False
        if self.adv:
            self.adv_label_dir = cfg.ADV_LABEL_DIR
            self.adv_vocab_file = cfg.ADV_VOCAB_FILE
            self.adv_label = self.get_adv_labels()
            self.adv_vocab = self.get_adv_vocab()
        self.target = True if cfg.TARGET_LABEL_DIR else False
        if self.target:
            self.target_label_dir = cfg.TARGET_LABEL_DIR
            self.target_label = self.get_target_labels()
            self.remap_labels = {self.target_label[i]: i for i in range(len(self.target_label))}
        # if os.path.exists(self.preprocessed):
        #     with open(self.preprocessed, "rb") as f:
        #         preprocessed = pickle.load(f)
        #         train = preprocessed["train"]
        #         test = preprocessed["test"]
        # else:
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = self.read_classnames(text_file)
        train = self.read_data(classnames, "train")
        # Follow standard practice to perform evaluation on the val set
        # Also used as the val set (so evaluate the last-step model)
        test = self.read_data(classnames, "test")

        preprocessed = {"train": train, "test": test}
        with open(self.preprocessed, "wb") as f:
            pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=test, test=test)

    def get_adv_labels(self):
        with open(os.path.join(self.adv_label_dir, 'instance_level_adv_label.json'), 'r') as f:
            adv_labels = json.load(f)
        return adv_labels

    def get_adv_vocab(self):
        with open(self.adv_vocab_file, 'r') as f:
            adv_vocab = f.readlines()
        adv_vocab = [cn.strip() for cn in adv_vocab]
        return adv_vocab

    def get_target_labels(self):
        with open(os.path.join(self.target_label_dir, 'target_label.json'), 'r') as f:
            target_labels = json.load(f)
        return target_labels

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                if self.adv:
                    if impath in self.adv_label:
                        adv_cn = self.adv_label[impath][0]
                    else:
                        adv_cn = random.choice(self.adv_vocab)
                    item = AdvDatum(impath=impath, label=label, classname=classname, adv_cn=adv_cn)
                elif self.target:
                    if classname not in self.target_label:
                        continue
                    label = self.remap_labels[classname]
                    item = Datum(impath=impath, label=label, classname=classname)
                else:
                    item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
