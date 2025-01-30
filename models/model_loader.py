import torch.nn as nn
from models.resnet import resnet
from models.densenet import densenet
from transformers import BertModel

def load_model(model_name, input_dim, num_classes):
    if model_name == "resnet":
        model = resnet(input_dim, num_classes)
    elif model_name == "densenet":
        model = densenet(input_dim, num_classes)
    elif model_name == "transformer":
        model = BertModel.from_pretrained("bert-base-uncased")
        model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    else:
        raise ValueError("Invalid model name")

    return model