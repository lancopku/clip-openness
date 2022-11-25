import torch
import os
from .dump_features import dump_pretrained_text_features
from utils.utils import model_bcakbone_name2model_name


def repe(dataset_dir, model_backbone, classnames, text_features, shift_lambda, retrieved_num):
    model_name = model_bcakbone_name2model_name(model_backbone)
    feature_dir = os.path.join(dataset_dir, f'{model_name}_pretrained_text_features_grouped_filtered.pt')
    if not os.path.exists(feature_dir):
        print('dump retrieved text features')
        dump_pretrained_text_features(dataset_dir, classnames, model_name)
    print(f"load retrieved text features from {feature_dir}")
    retrieved_features = torch.load(feature_dir)
    for i, cn in enumerate(classnames):
        cur_retrieved_features = retrieved_features[cn][:retrieved_num]
        # norm then average
        cur_retrieved_features /= cur_retrieved_features.norm(dim=-1, keepdim=True)
        avg_retrieved_features = torch.mean(cur_retrieved_features, dim=0)
        text_features[i] = (1 - shift_lambda) * text_features[i] + shift_lambda * avg_retrieved_features
    return text_features
