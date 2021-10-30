import torch
import matplotlib.pyplot as plt
from fuslib import utils
from PIL import Image

def trans_cae(model, conf):
    teckel_img = utils.normalize_im(Image.open(conf.data_dir_test + "teckel.jpg"), conf)
    teckel_img = torch.unsqueeze(teckel_img, dim=0)
    print(teckel_img.shape)
    print(model)
    im = model(teckel_img)
    im = im.detach().numpy()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12,4))
    ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[])
    utils.show_im(teckel_img[0])
    ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
    utils.show_im(im[0])
    plt.show()