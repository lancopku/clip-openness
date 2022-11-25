import os
import pickle
from collections import OrderedDict
import json
from robustness.tools.breeds_helpers import make_entity13, ClassHierarchy
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden


@DATASET_REGISTRY.register()
class ImageNet_Entity13(DatasetBase):
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
        ret = make_entity13(info_dir, split=None)
        _, subclass_split, label_map = ret
        self.superclass2class = {}
        for i, superclass in enumerate(label_map.values()):
            self.superclass2class[superclass] = [original_simple_classnames_mapping[hier.LEAF_NUM_TO_NAME[_id]] for _id
                                                 in subclass_split[0][i]]
        self.class2superclass = {v_i: k for k, v in self.superclass2class.items() for v_i in v}
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.target = True if cfg.TARGET_LABEL_DIR else False
        if self.target:
            self.target_label_dir = cfg.TARGET_LABEL_DIR
            self.target_label = self.get_target_labels()
            self.remap_labels = {self.target_label[i]: i for i in range(len(self.target_label))}

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = self.read_classnames(text_file)
        train = self.read_data(classnames, "train")
        # Follow standard practice to perform evaluation on the val set
        # Also used as the val set (so evaluate the last-step model)
        test = self.read_data(classnames, "val")

        preprocessed = {"train": train, "test": test}
        with open(self.preprocessed, "wb") as f:
            pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

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

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                if self.target:
                    if classname not in self.target_label:
                        continue
                    label = self.remap_labels[classname]
                    item = Datum(impath=impath, label=label, classname=classname)
                else:
                    item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items


# entity13
# superclass2class = {
#     'garment': ['trench coat', 'abaya', 'gown', 'poncho', 'military uniform', 'T-shirt', 'cloak', 'bikini', 'miniskirt',
#                 'swim trunks / shorts', 'lab coat', 'bra', 'hoop skirt', 'cardigan', 'pajamas', 'academic gown',
#                 'apron', 'diaper', 'sweatshirt', 'sarong'],
#     'bird': ['african grey parrot', 'bee eater', 'coucal', 'American coot', 'indigo bunting', 'king penguin',
#              'spoonbill', 'limpkin', 'quail', 'kite (bird of prey)', 'prairie grouse', 'red-breasted merganser',
#              'albatross', 'American dipper', 'goose', 'oystercatcher', 'great egret', 'hen', 'lorikeet',
#              'ruffed grouse'],
#     'reptile, reptilian': ['Gila monster', 'agama', 'triceratops', 'chameleon', 'worm snake', 'Indian cobra',
#                            'smooth green snake', 'mud turtle', 'water snake', 'loggerhead sea turtle',
#                            'sidewinder rattlesnake', 'leatherback sea turtle', 'boa constrictor', 'garter snake',
#                            'terrapin', 'box turtle', 'ring-necked snake', 'African rock python', 'Carolina anole',
#                            'European green lizard'],
#     'arthropod': ['rock crab', 'yellow garden spider', 'tiger beetle', 'southern black widow', 'barn spider',
#                   'leafhopper', 'ground beetle', 'fiddler crab', 'bee', 'stick insect', 'small white butterfly',
#                   'red admiral butterfly', 'lacewing', 'trilobite', 'sulphur butterfly', 'cicada',
#                   'European garden spider', 'leaf beetle', 'longhorn beetle', 'fly'],
#     'mammal, mammalian': ['Siamese cat', 'Alpine ibex', 'tiger', 'hippopotamus', 'Norwegian Elkhound', 'dugong',
#                           'black-and-white colobus', 'Samoyed', 'Persian cat', 'Irish Wolfhound', 'English Setter',
#                           'llama', 'red panda', 'armadillo', 'indri', 'Giant Schnauzer', 'pug', 'Dobermann',
#                           'American Staffordshire Terrier', 'Beagle'],
#     'accessory, accoutrement, accouterment': ['baby bib', 'feather boa', 'scarf', 'plastic bag', 'swimming cap',
#                                               'cowboy boot', 'necklace', 'crash helmet', 'gas mask or respirator',
#                                               'tights', 'hair clip', 'umbrella', 'Pickelhaube', 'mitten', 'sombrero',
#                                               'shower cap', 'sock', 'sneaker', 'graduation cap', 'handkerchief'],
#     'craft': ['catamaran', 'motorboat', 'fireboat', 'sailboat', 'airliner', 'container ship', 'ocean liner', 'trimaran',
#               'space shuttle', 'aircraft carrier', 'schooner', 'gondola', 'canoe', 'shipwreck', 'military aircraft',
#               'balloon', 'submarine', 'pirate ship', 'lifeboat', 'airship'],
#     'equipment': ['volleyball', 'notebook computer', 'basketball', 'hand-held computer', 'tripod', 'projector',
#                   'barbell', 'monitor', 'croquet ball', 'balance beam', 'cassette player', 'snorkel',
#                   'gymnastic horizontal bar', 'soccer ball', 'racket', 'baseball', 'joystick', 'microphone',
#                   'tape player', 'reflex camera'],
#     'furniture, piece of furniture, article of furniture': ['wardrobe', 'toilet seat', 'filing cabinet', 'mosquito net',
#                                                             'four-poster bed', 'bassinet', 'chiffonier',
#                                                             'folding chair', 'fire screen',
#                                                             'shoji screen / room divider', 'couch', 'throne',
#                                                             'infant bed', 'rocking chair', 'dining table', 'park bench',
#                                                             'storage chest', 'window screen', 'medicine cabinet',
#                                                             'barber chair'],
#     'instrument': ['upright piano', 'padlock', 'lighter', 'steel drum', 'parking meter', 'cleaver', 'syringe', 'abacus',
#                    'weighing scale', 'corkscrew', 'maraca', 'salt shaker', 'magnetic compass', 'accordion',
#                    'digital clock', 'screw', 'can opener', 'odometer', 'pipe organ', 'screwdriver'],
#     'man-made structure, construction': ['castle', 'bell tower', 'fountain', 'planetarium', 'traffic light',
#                                          'breakwater', 'cliff dwelling', 'monastery', 'prison', 'water tower',
#                                          'suspension bridge', 'split-rail fence', 'turnstile', 'tile roof',
#                                          'lighthouse', 'traffic or street sign', 'maze', 'chain-link fence', 'bakery',
#                                          'drilling rig'],
#     'wheeled vehicle': ['snowplow', 'semi-trailer truck', 'race car', 'shopping cart', 'unicycle', 'vespa',
#                         'railroad car', 'minibus', 'jeep', 'recreational vehicle', 'rickshaw', 'golf cart', 'tow truck',
#                         'ambulance', 'high-speed train', 'fire truck', 'horse-drawn vehicle', 'tram', 'tank',
#                         'ford model t'],
#     'produce, green goods, green groceries, garden truck': ['broccoli', 'corn', 'orange', 'cucumber',
#                                                             'spaghetti squash', 'butternut squash', 'acorn squash',
#                                                             'cauliflower', 'bell pepper', 'fig', 'pomegranate',
#                                                             'mushroom', 'strawberry', 'lemon', 'cabbage',
#                                                             'Granny Smith apple', 'rose hip', 'corn cob', 'banana',
#                                                             'artichoke']}
