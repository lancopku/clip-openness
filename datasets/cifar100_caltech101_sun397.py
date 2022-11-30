import os
import pickle
import copy
from collections import OrderedDict
import numpy as np
import random
import json
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from .adv_label_base_dataset import AdvDatum
from dassl.utils import read_json, write_json
from datasets import caltech101, sun397, my_cifar100


@DATASET_REGISTRY.register()
class CIFAR100_CALTECH101_SUN397(DatasetBase):
    dataset_dir = "cifar100_caltech101_sun397"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.superclass2class = {
            'MYCIFAR100': ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                           'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                           'caterpillar',
                           'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
                           'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                           'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                           'lobster',
                           'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                           'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                           'poppy',
                           'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                           'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                           'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                           'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                           'worm'],
            'Caltech101': ['flamingo', 'mayfly', 'bonsai', 'car_side', 'tick', 'butterfly', 'garfield', 'sunflower',
                           'watch', 'barrel', 'cougar_face', 'inline_skate', 'cellphone', 'dolphin', 'electric_guitar',
                           'rhino', 'face', 'stegosaurus', 'elephant', 'stapler', 'airplane', 'umbrella', 'buddha',
                           'crayfish', 'scissors', 'crocodile', 'menorah', 'pizza', 'llama', 'dollar_bill', 'cup',
                           'pigeon', 'gerenuk', 'ewer', 'ferry', 'strawberry', 'ant', 'crab', 'pyramid', 'motorbike',
                           'soccer_ball', 'cougar_body', 'laptop', 'dragonfly', 'schooner', 'okapi', 'starfish',
                           'revolver', 'helicopter', 'emu', 'windsor_chair', 'lotus', 'sea_horse', 'headphone',
                           'flamingo_head', 'minaret', 'binocular', 'trilobite', 'beaver', 'saxophone', 'platypus',
                           'crocodile_head', 'kangaroo', 'euphonium', 'ceiling_fan', 'gramophone', 'metronome',
                           'pagoda', 'yin_yang', 'brontosaurus', 'ketch', 'nautilus', 'bass', 'joshua_tree', 'cannon',
                           'octopus', 'panda', 'lamp', 'ibis', 'chair', 'scorpion', 'hedgehog', 'wheelchair',
                           'water_lilly', 'wrench', 'camera', 'mandolin', 'snoopy', 'hawksbill', 'stop_sign',
                           'leopard', 'anchor', 'grand_piano', 'chandelier', 'accordion', 'dalmatian', 'lobster',
                           'brain', 'wild_cat', 'rooster'],
            'SUN397': ['ballroom', 'outdoor church', 'windmill', 'bathroom', 'courtroom',
                       'subway_interior', 'ruin', 'batters_box', 'lift_bridge', 'lock_chamber',
                       'parlor', 'thriftshop', 'ball_pit', 'bar', 'abbey', 'picnic_area',
                       'indoor swimming_pool', 'backseat car_interior', 'water_tower', 'street',
                       'snowfield', 'bedroom', 'swamp', 'parking_lot', 'ice_cream_parlor',
                       'iceberg', 'courtyard', 'utility_room', 'frontseat car_interior',
                       'indoor warehouse', 'galley', 'landfill', 'outdoor control_tower',
                       'cottage_garden', 'herb_garden', 'conference_center', 'tree_farm', 'office',
                       'indoor firing_range', 'outdoor monastery', 'hot_spring', 'bowling_alley',
                       'indoor_seats theater', 'house', 'indoor_procenium theater', 'boat_deck',
                       'physics_laboratory', 'home poolroom', 'elevator_shaft', 'airplane_cabin',
                       'dining_car', 'arch', 'south_asia temple', 'corridor', 'heliport',
                       'outdoor tennis_court', 'cheese_factory', 'slum', 'putting_green',
                       'music_store', 'coffee_shop', 'art_school', 'pulpit', 'basement',
                       'sea_cliff', 'mountain', 'bullring', 'ocean', 'home_office',
                       'fastfood_restaurant', 'wind_farm', 'beach', 'fountain',
                       'exterior covered_bridge', 'laundromat', 'outdoor power_plant',
                       'outdoor bazaar', 'baseball_field', 'sandbox', 'rope_bridge', 'pharmacy',
                       'bottle_storage wine_cellar', 'baseball stadium', 'vegetable_garden',
                       'chalet', 'indoor market', 'outdoor greenhouse', 'indoor brewery',
                       'outdoor hunting_lodge', 'biology_laboratory', 'archive', 'crosswalk',
                       'wet_bar', 'stable', 'banquet_hall', 'burial_chamber', 'indoor stage',
                       'golf_course', 'reception', 'lobby', 'tower', 'nursery', 'veranda', 'marsh',
                       'pagoda', 'hotel_room', 'canyon', 'skatepark', 'train_railway',
                       'auto_factory', 'residential_neighborhood', 'hospital_room',
                       'indoor florist_shop', 'restaurant_kitchen', 'driveway',
                       'indoor tennis_court', 'shower', 'indoor library', 'dorm_room',
                       'indoor kennel', 'discotheque', 'motel', 'volcano', 'mausoleum', 'orchard',
                       'outdoor doorway', 'aqueduct', 'door elevator', 'dock', 'railroad_track',
                       'outdoor hotel', 'plaza', 'interior balcony', 'urban canal', 'chemistry_lab',
                       'indoor church', 'sandbar', 'kitchenette', 'indoor diner',
                       'indoor greenhouse', 'pond', 'promenade_deck', 'amusement_park', 'mansion',
                       'forest_road', 'basilica', 'living_room', 'east_asia temple',
                       'indoor pilothouse', 'amusement_arcade', 'clean_room', 'vineyard',
                       'recreation_room', 'delicatessen', 'indoor garage', 'ski_resort', 'berth',
                       'indoor badminton_court', 'hill', 'rice_paddy', 'music_studio', 'excavation',
                       'catacomb', 'rock_arch', 'server_room', 'control_room',
                       'outdoor parking_garage', 'raceway', 'lecture_room', 'raft',
                       'indoor cathedral', 'indoor factory', 'ice_floe', 'needleleaf forest',
                       'outdoor synagogue', 'gas_station', 'indoor bow_window', 'anechoic_chamber',
                       'indoor movie_theater', 'closet', 'fan waterfall', 'indoor cloister',
                       'indoor ice_skating_rink', 'creek', 'platform subway_station', 'locker_room',
                       'barrel_storage wine_cellar', 'indoor shopping_mall', 'indoor pub',
                       'bus_interior', 'phone_booth', 'outdoor driving_range', 'classroom',
                       'indoor volleyball_court', 'indoor jail', 'sauna', 'wild field', 'cockpit',
                       'gift_shop', 'riding_arena', 'indoor wrestling_ring', 'indoor booth',
                       'village', 'manufactured_home', 'shop bakery', 'electrical_substation',
                       'pantry', 'trench', 'formal_garden', 'outdoor lido_deck', 'van_interior',
                       'highway', 'natural lake', 'indoor jacuzzi', 'clothing_store',
                       'outdoor podium', 'cliff', 'jewelry_shop', 'outdoor arrival_gate',
                       'outdoor swimming_pool', 'bookstore', 'conference_room', 'hospital',
                       'schoolhouse', 'building_facade', 'assembly_line', 'outdoor diner',
                       'pavilion', 'attic', 'restaurant', 'fire_escape', 'wheat_field',
                       'squash_court', 'forest_path', 'kindergarden_classroom', 'home dinette',
                       'outdoor track', 'food_court', 'videostore', 'game_room', 'indoor hangar',
                       'medina', 'kasbah', 'indoor mosque', 'indoor casino', 'water moat', 'coast',
                       'hayfield', 'indoor synagogue', 'dam', 'alley', 'waiting_room',
                       'outdoor tent', 'bayou', 'campus', 'outdoor hangar', 'vegetation desert',
                       'outdoor oil_refinery', 'ski_slope', 'wave', 'outdoor outhouse',
                       'dentists_office', 'patio', 'plunge waterfall', 'public atrium', 'badlands',
                       'outdoor bow_window', 'outdoor hot_tub', 'lighthouse', 'ice_shelf',
                       'throne_room', 'rainforest', 'cafeteria', 'outdoor labyrinth',
                       'garbage_dump', 'pasture', 'outdoor cabin', 'igloo', 'restaurant_patio',
                       'outdoor athletic_field', 'baggage_claim', 'ski_lodge', 'corn_field',
                       'outdoor kennel', 'ticket_booth', 'shoe_shop', 'outdoor nuclear_power_plant',
                       'harbor', 'natural canal', 'playground', 'block waterfall', 'racecourse',
                       'indoor bazaar', 'establishment poolroom', 'supermarket', 'barn',
                       'indoor bistro', 'indoor escalator', 'viaduct', 'oast_house', 'sand desert',
                       'indoor cavern', 'airport_terminal', 'fairway', 'skyscraper', 'engine_room',
                       'indoor parking_garage', 'landing_deck', 'sushi_bar', 'indoor apse',
                       'outdoor inn', 'outdoor chicken_coop', 'campsite', 'butte', 'computer_room',
                       'shopfront', 'beauty_salon', 'river', 'courthouse', 'botanical_garden',
                       'outdoor general_store', 'outdoor mosque', 'outdoor volleyball_court',
                       'oilrig', 'drugstore', 'aquarium', 'indoor museum', 'industrial_area',
                       'toyshop', 'toll_plaza', 'office cubicle', 'crevasse',
                       'outdoor apartment_building', 'auditorium', 'boardwalk', 'boxing_ring',
                       'indoor podium', 'playroom', 'shed', 'indoor gymnasium', 'park',
                       'fire_station', 'construction_site', 'art_gallery', 'coral_reef underwater',
                       'bridge', 'outdoor library', 'kitchen', 'palace', 'jail_cell', 'tree_house',
                       'butchers_shop', 'broadleaf forest', 'exterior balcony', 'amphitheater',
                       'runway', 'vehicle dinette', 'dining_room', 'outdoor basketball_court',
                       'indoor chicken_coop', 'interior elevator', 'television_studio',
                       'mountain_snowy', 'operating_room', 'candy_store', 'sky', 'staircase',
                       'outdoor ice_skating_rink', 'limousine_interior', 'football stadium',
                       'carrousel', 'outdoor market', 'bamboo_forest', 'indoor general_store',
                       'childs_room', 'exterior gazebo', 'boathouse', 'cultivated field',
                       'martial_arts_gym', 'islet', 'outdoor cathedral', 'youth_hostel', 'cemetery',
                       'watering_hole', 'veterinarians_office', 'platform train_station',
                       'outdoor observatory', 'art_studio', 'corral', 'topiary_garden', 'yard',
                       'castle', 'valley', 'outdoor planetarium', 'barndoor', 'office_building',
                       'fishpond']}
        self.class2superclass = {v_i: k for k, v in self.superclass2class.items() for v_i in v}
        self.vocab_num2dataset_num = {
            'MYCIFAR100': 'cifar100',
            'Caltech101': 'caltech-101',
            'SUN397': 'sun397'
        }

        dup_cfg = copy.deepcopy(cfg)
        dup_cfg.TARGET_LABEL_DIR = None
        cifar100_dataset = my_cifar100.MYCIFAR100(dup_cfg)
        assert cifar100_dataset.num_classes == len(self.superclass2class['MYCIFAR100'])
        caltech101_dataset = caltech101.Caltech101(cfg)
        assert caltech101_dataset.num_classes == len(self.superclass2class['Caltech101'])
        sun397_dataset = sun397.SUN397(cfg)
        assert sun397_dataset.num_classes == len(self.superclass2class['SUN397'])

        train = cifar100_dataset.train_x + caltech101_dataset.train_x + sun397_dataset.train_x
        val = cifar100_dataset.val + caltech101_dataset.val + sun397_dataset.val
        test = cifar100_dataset.test + caltech101_dataset.test + sun397_dataset.test

        self.adv = True if cfg.ADV_LABEL_DIR and cfg.ADV_VOCAB_FILE else False
        if self.adv:
            self.adv_label_dir = cfg.ADV_LABEL_DIR
            self.adv_vocab_file = cfg.ADV_VOCAB_FILE
            self.adv_label = self.get_adv_labels()
            self.adv_vocab = self.get_adv_vocab()
        self.target = True if cfg.TARGET_LABEL_DIR else False
        if self.target:
            self.target_label_dir = cfg.TARGET_LABEL_DIR
            self.target_label = list(set(self.get_target_labels()))
            self.remap_labels = {self.target_label[i]: i for i in range(len(self.target_label))}

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        if self.target:
            train = self.filter_data(train, cfg)
            val = self.filter_data(val, cfg)
            test = self.filter_data(test, cfg)

        super().__init__(train_x=train, val=val, test=test)

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

    def filter_data(self, data, cfg):
        filtered_data = []
        if not hasattr(cfg, 'VOCABS'):
            cfg.VOCABS = list(self.superclass2class.keys())
        for datum in data:
            if datum.classname in self.target_label and any(
                    [self.vocab_num2dataset_num[v] in datum.impath for v in cfg.VOCABS]):
                label = self.remap_labels[datum.classname]
                filtered_data.append(Datum(impath=datum.impath, label=label, classname=datum.classname))
        return filtered_data
