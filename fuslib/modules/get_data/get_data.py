from PIL import Image
from os import listdir
import pickle
import torch
from fuslib.utils import normalize_im

def get_data(conf):
    print("Get the images...")

    if(conf.store):
        data = []
        for i, f in enumerate(listdir(conf.data_dir_train)):
            if conf.verbose:
                print(str(i) + " / " + str(len(listdir(conf.data_dir_train))))
            data.append(normalize_im(Image.open(conf.data_dir_train + f), conf))
            if i == conf.nb_images - 1:
                break

        data = torch.stack(data)

        with open(conf.work_dir + '/data/data', 'wb') as f1:
            pickle.dump(data, f1)

    else:
        with open(conf.work_dir + '/data/data', 'rb') as f1:
            data = pickle.load(f1)

    if conf.verbose:
        print(data.shape)

    print("Get the images done.\n")

    return data
   