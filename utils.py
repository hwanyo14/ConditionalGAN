import torch
from torch import nn
import torch.nn.functional as F


# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_result_images(gen, latent, label, num_class, device):
    gen.eval()

    encoded_label = F.one_hot(label, num_classes=num_class).to(device)

    with torch.no_grad():
        generated = gen(latent, encoded_label)

    generated = generated * 0.5 + 0.5

    return generated