import os
import pickle
from collections import OrderedDict
import json
from utils.breeds_helpers import make_living17, ClassHierarchy
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
from tqdm import tqdm


@DATASET_REGISTRY.register()
class ImageNet_Living17(DatasetBase):
    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # "A subset of these class names are modified from the default ImageNet class names sourced from Anish Athalye's imagenet-simple-labels."
        # https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
        with open(os.path.join(self.dataset_dir, 'original_simple_classnames_mapping.json'), 'r') as f:
            original_simple_classnames_mapping = json.load(f)

        info_dir = os.path.join(self.dataset_dir, 'modified')
        hier = ClassHierarchy(info_dir)
        ret = make_living17(info_dir, split=None)
        _, subclass_split, label_map = ret
        self.superclass2class = {}
        for i, superclass in enumerate(label_map.values()):
            self.superclass2class[superclass] = [original_simple_classnames_mapping[hier.LEAF_NUM_TO_NAME[_id]] for _id
                                                 in subclass_split[0][i]]
        self.class2superclass = {v_i: k for k, v in self.superclass2class.items() for v_i in v}
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.target = True if cfg.TARGET_LABEL_DIR else False

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.target:
            self.target_label_dir = cfg.TARGET_LABEL_DIR
            self.target_label = self.get_target_labels()
            self.remap_labels = {self.target_label[i]: i for i in range(len(self.target_label))}
            train = self.post_process(train)
            test = self.post_process(test)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=test, test=test)

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

        for label, folder in tqdm(enumerate(folders)):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def post_process(self, items):
        processed_items = []
        for item in items:
            if item.classname not in self.target_label:
                continue
            processed_label = self.remap_labels[item.classname]
            processed_item = Datum(impath=item.impath, label=processed_label, classname=item.classname)
            processed_items.append(processed_item)
        return processed_items
# living17
# superclass2class = {
#     'salamander': ['newt', 'axolotl', 'smooth newt', 'spotted salamander'],
#     'turtle': ['box turtle', 'leatherback sea turtle', 'loggerhead sea turtle', 'mud turtle'],
#     'lizard': ['desert grassland whiptail lizard', 'alligator lizard', 'chameleon', 'banded gecko'],
#     'snake, serpent, ophidian': ['night snake', 'garter snake', 'sea snake', 'boa constrictor'],
#     'spider': ['tarantula', 'yellow garden spider', 'European garden spider', 'wolf spider'],
#     'grouse': ['ptarmigan', 'prairie grouse', 'ruffed grouse', 'black grouse'],
#     'parrot': ['macaw', 'lorikeet', 'african grey parrot', 'sulphur-crested cockatoo'],
#     'crab': ['Dungeness crab', 'fiddler crab', 'rock crab', 'red king crab'],
#     'dog, domestic dog, Canis familiaris': ['Bloodhound', 'Pekingese', 'Great Pyrenees dog', 'Papillon'],
#     'wolf': ['coyote', 'red wolf or maned wolf', 'Alaskan tundra wolf', 'grey wolf'],
#     'fox': ['grey fox', 'Arctic fox', 'red fox', 'kit fox'],
#     'domestic cat, house cat, Felis domesticus, Felis catus': ['tiger cat', 'Egyptian Mau', 'Persian cat',
#                                                                'Siamese cat'],
#     'bear': ['sloth bear', 'American black bear', 'polar bear', 'brown bear'],
#     'beetle': ['dung beetle', 'rhinoceros beetle', 'ground beetle', 'longhorn beetle'],
#     'butterfly': ['sulphur butterfly', 'red admiral butterfly', 'small white butterfly', 'ringlet butterfly'],
#     'ape': ['gibbon', 'orangutan', 'gorilla', 'chimpanzee'],
#     'monkey': ['marmoset', 'titi monkey', "Geoffroy's spider monkey", 'howler monkey']
# }

