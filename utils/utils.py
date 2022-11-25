def cifar10_classes():
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return cifar10_classes


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


def cifar100_templates():
    CIFAR100_TEMPLATES = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
        "a photo of the {}.",
        "a blurry photo of the {}.",
        "a black and white photo of the {}.",
        "a low contrast photo of the {}.",
        "a high contrast photo of the {}.",
        "a bad photo of the {}.",
        "a good photo of the {}.",
        "a photo of the small {}.",
        "a photo of the big {}."
    ]
    return CIFAR100_TEMPLATES


def model_bcakbone_name2model_name(model_backbone):
    model_bcakbone_name2model_name = {
        'ViT-B/32': 'clip_vit_b32',
        'ViT-B/16': 'clip_vit_b16',
        'ViT-B/32/DeClip': 'declip_vit_b32',
        "ViT-B/16/ep25": 'slip_vit_b16_ep25',
        "ViT-B/16/ep50": 'slip_vit_b16_ep50',
        "ViT-B/16/ep100": 'slip_vit_b16_ep100'
    }
    return model_bcakbone_name2model_name[model_backbone]
