import torch

def preproc(data, conf):
    print("Preprocessing of the data...")

    data = torch.utils.data.DataLoader(data, batch_size=conf.batch_size, num_workers=0)

    print("Preprocessing of the data done.\n")

    return data
