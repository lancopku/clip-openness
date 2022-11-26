from clip_retrieval import ClipClient, Modality
import json
import os
# from utils.utils import cifar100_classes, imagenet_classes
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def cifar100_classes():
    cifar100_classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark',
                        'trout',
                        'orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'bottle', 'bowl', 'can', 'cup', 'plate',
                        'apple',
                        'mushroom', 'orange', 'pear', 'sweet_pepper', 'clock', 'keyboard', 'lamp', 'telephone',
                        'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly',
                        'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle',
                        'house',
                        'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle',
                        'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab',
                        'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile',
                        'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                        'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree', 'bicycle', 'bus',
                        'motorcycle',
                        'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    return cifar100_classes


def imagenet_classes():
    imagenet_classes = []
    with open('data/imagenet/classnames.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            imagenet_classes.append(classname)
    return imagenet_classes

dataset = 'imagenet'
client = ClipClient(url="https://knn5.laion.ai/knn-service", indice_name="laion5B", num_images=20000)

if dataset == 'imagenet':
    class_list = imagenet_classes()
elif dataset == 'cifar100':
    class_list = cifar100_classes()
else:
    raise ValueError('not support dataset')

folder = f"data/{dataset}/laion5B_retrieval_1000"
if not os.path.exists(folder):
    os.mkdir(folder)

for class_name in class_list:
    file_name = os.path.join(folder, str(class_name).replace(' / ', ' or ') + ".json")
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            res = f.readlines()
            if len(res) > 0:
                continue
    f = open(file_name, "w", encoding="UTF-8")
    text = "a photo of a %s" % class_name.replace('_', ' ')
    print(text)
    results = client.query(text=text)
    # print(results[0].keys())
    # print(len(results))
    length = min(len(results), 1000)
    json.dump(results[:length], f, ensure_ascii=False)
    print("retrieve %d pairs for %s" % (length, class_name))

