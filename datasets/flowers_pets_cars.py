import os
import pickle
from collections import OrderedDict
import numpy as np
import random
import json
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from .adv_label_base_dataset import AdvDatum
from dassl.utils import read_json, write_json
from datasets import oxford_pets, oxford_flowers, stanford_cars


@DATASET_REGISTRY.register()
class FLOWERS_PETS_CARS(DatasetBase):
    dataset_dir = "flowers_pets_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.superclass2class = {
            'OxfordFlowers': ['king protea', 'hippeastrum', 'bee balm', 'artichoke', 'bird of paradise', 'watercress',
                               'great masterwort', 'peruvian lily', 'english marigold', 'siam tulip', 'geranium',
                               'windflower', 'red ginger', 'magnolia', 'mexican petunia', 'sword lily', 'anthurium',
                               'hard-leaved pocket orchid', 'petunia', 'marigold', 'blanket flower', 'stemless gentian',
                               'garden phlox', 'monkshood', 'grape hyacinth', 'cyclamen', 'love in the mist',
                               'cape flower', 'canna lily', 'rose', 'osteospermum', 'tree poppy', 'azalea',
                               'silverbush', 'gaura', 'bolero deep blue', 'sweet william', 'poinsettia',
                               'alpine sea holly', 'yellow iris', 'frangipani', 'common dandelion', 'oxeye daisy',
                               'tiger lily', 'orange dahlia', 'trumpet creeper', 'clematis', 'wallflower',
                               'giant white arum lily', 'foxglove', 'camellia', 'thorn apple', 'blackberry lily',
                               'sweet pea', 'fire lily', 'bearded iris', 'pelargonium', 'pink-yellow dahlia',
                               'toad lily', 'wild pansy', 'fritillary', 'ruby-lipped cattleya', 'spring crocus',
                               'purple coneflower', "colt's foot", 'barbeton daisy', 'prince of wales feathers',
                               'sunflower', 'carnation', 'canterbury bells', 'mallow', 'bougainvillea', 'daffodil',
                               'mexican aster', 'snapdragon', 'corn poppy', 'pincushion flower', 'morning glory',
                               'californian poppy', 'globe-flower', 'hibiscus', 'japanese anemone', 'black-eyed susan',
                               'lotus', 'bishop of llandaff', 'passion flower', 'cautleya spicata', 'pink primrose',
                               'bromelia', 'buttercup', 'columbine', 'globe thistle', 'spear thistle', 'primula',
                               'moon orchid', 'water lily', 'gazania', 'balloon flower', 'lenten rose', 'tree mallow',
                               'ball moss', 'desert-rose'],
            'OxfordPets': ['egyptian_mau', 'newfoundland', 'pomeranian', 'birman', 'beagle', 'miniature_pinscher',
                            'abyssinian', 'english_cocker_spaniel', 'havanese', 'siamese', 'sphynx',
                            'staffordshire_bull_terrier', 'bombay', 'pug', 'american_bulldog', 'maine_coon',
                            'saint_bernard', 'bengal', 'russian_blue', 'boxer', 'great_pyrenees', 'yorkshire_terrier',
                            'american_pit_bull_terrier', 'ragdoll', 'basset_hound', 'leonberger', 'scottish_terrier',
                            'british_shorthair', 'shiba_inu', 'german_shorthaired', 'japanese_chin', 'english_setter',
                            'wheaten_terrier', 'chihuahua', 'keeshond', 'samoyed', 'persian'],
            'StanfordCars': ['2012 Chevrolet Camaro Convertible', '2012 McLaren MP4-12C Coupe',
                              '2012 Ferrari FF Coupe', '2012 BMW M3 Coupe', '2012 GMC Acadia SUV',
                              '2009 Chrysler Aspen SUV', '2012 Honda Odyssey Minivan', '2012 Ram C/V Cargo Van Minivan',
                              '1999 Plymouth Neon Coupe', '2007 Dodge Dakota Club Cab', '2009 Spyker C8 Coupe',
                              '2012 GMC Terrain SUV', '2002 Daewoo Nubira Wagon', '2012 Buick Verano Sedan',
                              '2007 Hyundai Elantra Sedan', '2012 Porsche Panamera Sedan',
                              '2012 Rolls-Royce Phantom Sedan', '2012 Ford F-450 Super Duty Crew Cab',
                              '2012 Toyota Corolla Sedan', '1994 Audi 100 Wagon',
                              '2012 Aston Martin V8 Vantage Convertible', '2012 Jeep Grand Cherokee SUV',
                              '2012 Suzuki SX4 Sedan', '2007 Bentley Continental GT Coupe',
                              '1998 Eagle Talon Hatchback', '2012 Dodge Caliber Wagon',
                              '2012 Hyundai Veloster Hatchback', '2012 Cadillac SRX SUV',
                              '2012 Chevrolet Avalanche Crew Cab', '2007 Chevrolet Malibu Sedan',
                              '2012 Nissan NV Passenger Van', '2012 Hyundai Accent Sedan',
                              '2012 MINI Cooper Roadster Convertible', '2011 Lincoln Town Car Sedan',
                              '2001 Lamborghini Diablo Coupe', '2012 Audi A5 Coupe', '2012 Land Rover LR2 SUV',
                              '2012 Acura ZDX Hatchback', '2009 Spyker C8 Convertible', '2012 BMW 1 Series Convertible',
                              '2012 Suzuki SX4 Hatchback', '2010 Chevrolet HHR SS', '2000 AM General Hummer SUV',
                              '2012 smart fortwo Convertible', '2011 Dodge Challenger SRT8',
                              '2012 Hyundai Sonata Hybrid Sedan', '2010 HUMMER H3T Crew Cab',
                              '2007 Bentley Continental Flying Spur Sedan', '2009 Mercedes-Benz SL-Class Coupe',
                              '2012 Dodge Durango SUV', '2012 Chevrolet Sonic Sedan', '2012 Dodge Journey SUV',
                              '2009 Bentley Arnage Sedan', '2012 Toyota 4Runner SUV',
                              '2008 Chrysler PT Cruiser Convertible', '1991 Volkswagen Golf Hatchback',
                              '2012 Volkswagen Beetle Hatchback', '2012 Volkswagen Golf Hatchback',
                              '2012 Toyota Sequoia SUV', '2012 Nissan Juke Hatchback', '2011 Ford Ranger SuperCab',
                              '2012 Jeep Compass SUV', '2010 Dodge Ram Pickup 3500 Crew Cab',
                              '2012 GMC Canyon Extended Cab', '2012 Cadillac CTS-V Sedan', '2012 BMW 3 Series Wagon',
                              '2009 Bugatti Veyron 16.4 Convertible', '2007 Honda Odyssey Minivan',
                              '2012 Mercedes-Benz E-Class Sedan', '2012 Aston Martin Virage Convertible',
                              '2007 Ford Mustang Convertible', '1998 Nissan 240SX Coupe',
                              '2012 Aston Martin V8 Vantage Coupe', '2012 Chevrolet Corvette Convertible',
                              '2012 Chevrolet Silverado 1500 Extended Cab', '2012 Dodge Charger Sedan',
                              '2012 FIAT 500 Convertible', '2012 Hyundai Sonata Sedan', '2012 Audi R8 Coupe',
                              '2007 Chevrolet Monte Carlo Coupe', '2012 BMW X3 SUV', '2008 Acura TL Type-S',
                              '2012 Scion xD Hatchback', '2012 Hyundai Elantra Touring Hatchback',
                              '2012 Chrysler Town and Country Minivan', '2012 Chevrolet Silverado 2500HD Regular Cab',
                              '2012 Buick Regal GS', '2009 Bugatti Veyron 16.4 Coupe', '2012 Audi S5 Convertible',
                              '2010 Chrysler Sebring Convertible', '2012 BMW X6 SUV', '2012 Acura TL Sedan',
                              '2012 Rolls-Royce Phantom Drophead Coupe Convertible', '2007 Volvo XC90 SUV',
                              '2012 Tesla Model S Sedan', '2012 Honda Accord Sedan',
                              '2012 Chevrolet Silverado 1500 Regular Cab', '2012 Lamborghini Aventador Coupe',
                              '2012 Jeep Liberty SUV', '2012 Nissan Leaf Hatchback', '2012 Hyundai Genesis Sedan',
                              '2012 Buick Enclave SUV', '2011 Audi TT Hatchback', '2007 Suzuki Aerio Sedan',
                              '2012 Lamborghini Gallardo LP 570-4 Superleggera', '2007 Buick Rainier SUV',
                              '2007 Chevrolet Impala Sedan', '2012 Chevrolet Corvette ZR1',
                              '2012 Aston Martin Virage Coupe', '2010 BMW M6 Convertible', '2010 Dodge Dakota Crew Cab',
                              '2012 Acura TSX Sedan', '2011 Infiniti QX56 SUV', '2011 Bentley Mulsanne Sedan',
                              '2008 Dodge Magnum Wagon', '2012 Mercedes-Benz S-Class Sedan',
                              '2012 Bentley Continental GT Coupe', '2012 Chevrolet Tahoe Hybrid SUV',
                              '2012 Ferrari 458 Italia Coupe', '2007 BMW X5 SUV', '1994 Audi 100 Sedan',
                              '2011 Audi S6 Sedan', '2012 Mitsubishi Lancer Sedan', '2012 BMW 1 Series Coupe',
                              '2012 Hyundai Azera Sedan', '2010 Chrysler 300 SRT-8', '2009 Dodge Charger SRT-8',
                              '2012 Toyota Camry Sedan', '1993 Mercedes-Benz 300-Class Convertible',
                              '2012 Audi S4 Sedan', '2009 Chevrolet TrailBlazer SS',
                              '2012 Ferrari 458 Italia Convertible', '2007 Cadillac Escalade EXT Crew Cab',
                              '2012 Rolls-Royce Ghost Sedan', '2009 Ford Expedition EL SUV', '2007 Audi S4 Sedan',
                              '2012 Suzuki Kizashi Sedan', '2008 Chrysler Crossfire Convertible', '2012 Audi TTS Coupe',
                              '2007 Chevrolet Silverado 1500 Classic Extended Cab', '2012 Land Rover Range Rover SUV',
                              '2012 FIAT 500 Abarth', '2008 Lamborghini Reventon Coupe', '2012 Ford E-Series Wagon Van',
                              '2007 Ford Freestar Minivan', '2012 GMC Savana Van', '2012 Jaguar XK XKR',
                              '2007 Ford F-150 Regular Cab', '2008 Isuzu Ascender SUV', '1993 Geo Metro Convertible',
                              '2012 Hyundai Veracruz SUV', '2012 Ford Edge SUV', '2012 BMW ActiveHybrid 5 Sedan',
                              '2007 Chevrolet Express Van', '2012 Mercedes-Benz Sprinter Van', '2007 Ford Focus Sedan',
                              '2012 Ferrari California Convertible', '2012 Jeep Wrangler SUV',
                              '2012 Honda Accord Coupe', '2008 Audi RS 4 Convertible',
                              '2010 Chevrolet Malibu Hybrid Sedan', '2012 Acura RL Sedan', '2012 Volvo C30 Hatchback',
                              '2012 GMC Yukon Hybrid SUV', '1993 Volvo 240 Sedan', '2001 Acura Integra Type R',
                              '2012 Hyundai Tucson SUV', '2007 Dodge Caliber Wagon', '2012 Fisker Karma Sedan',
                              '2010 BMW M5 Sedan', '2009 Dodge Ram Pickup 3500 Quad Cab', '2012 Ford F-150 Regular Cab',
                              '2010 Chevrolet Cobalt SS', '2012 Chevrolet Silverado 1500 Hybrid Crew Cab',
                              '1997 Dodge Caravan Minivan', '2012 Jeep Patriot SUV', '2012 Audi TT RS Coupe',
                              '2009 Dodge Sprinter Cargo Van', '2012 Infiniti G Coupe IPL', '2012 Ford Fiesta Sedan',
                              '2007 Chevrolet Corvette Ron Fellows Edition Z06', '2012 Hyundai Santa Fe SUV',
                              '2006 Ford GT Coupe', '2011 Mazda Tribute SUV',
                              '2012 Bentley Continental Supersports Conv. Convertible',
                              '2012 Maybach Landaulet Convertible', '2007 BMW 6 Series Convertible',
                              '2012 Audi S5 Coupe', '2012 Mercedes-Benz C-Class Sedan', '2007 Dodge Durango SUV',
                              '2009 HUMMER H2 SUT Crew Cab', '1994 Audi V8 Sedan', '2012 Chevrolet Traverse SUV',
                              '2007 Chevrolet Express Cargo Van', '2012 BMW Z4 Convertible', '2012 BMW 3 Series Sedan']}
        self.class2superclass = {v_i: k for k, v in self.superclass2class.items() for v_i in v}

        oxfordflowers = oxford_flowers.OxfordFlowers(cfg)
        assert oxfordflowers.num_classes == len(self.superclass2class['OxfordFlowers'])
        oxfordpets = oxford_pets.OxfordPets(cfg)
        assert oxfordpets.num_classes == len(self.superclass2class['OxfordPets'])
        stanfordcars = stanford_cars.StanfordCars(cfg)
        assert stanfordcars.num_classes == len(self.superclass2class['StanfordCars'])

        train = oxfordflowers.train_x + oxfordpets.train_x + stanfordcars.train_x
        val = oxfordflowers.val + oxfordpets.val + stanfordcars.val
        test = oxfordflowers.test + oxfordpets.test + stanfordcars.test

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

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        if self.target:
            train = self.filter_data(train)
            val = self.filter_data(val)
            test = self.filter_data(test)

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

    def filter_data(self, data):
        filtered_data = []
        for datum in data:
            if datum.classname not in self.target_label:
                continue
            label = self.remap_labels[datum.classname]
            filtered_data.append(Datum(impath=datum.impath, label=label, classname=datum.classname))
        return filtered_data
