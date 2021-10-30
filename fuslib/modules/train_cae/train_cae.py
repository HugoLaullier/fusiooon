from fuslib import models
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

def plot_latent(autoencoder, data):
    for i, x in enumerate(data):
        z = autoencoder.encoder(x)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], cmap='tab10')
        plt.show()


def train_cae(data, conf):
    print("Training Convolutionnal AutoEncoder...")
    if conf.train:
        #Instantiate the model
        model = getattr(models, conf.model)    
        mdl = model(conf)
        if conf.verbose:
            print(mdl)

        # plot_latent(mdl, data)
        # exit()

        #Optimizer
        optim_id = getattr(torch.optim,conf.optim)
        optim = optim_id(mdl.parameters(), lr=conf.lr)

        for epoch in range(1, conf.nb_epochs+1):
            # monitor training loss
            train_loss = 0.0

            #Training
            for images in data:
                optim.zero_grad()
                outputs = mdl(images)
                loss = conf.criterion(outputs, images)
                loss.backward()
                optim.step()
                train_loss += loss.item()*images.size(0)
                
            train_loss = train_loss/len(data)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        with open(conf.work_dir + '/model', 'wb') as f:
            pickle.dump(mdl, f)
    else:
        with open(conf.work_dir + '/model', 'rb') as f:
            mdl = pickle.load(f)

    print("Training Convolutionnal AutoEncoder done.\n")
    
    return mdl