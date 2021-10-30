import matplotlib.pyplot as plt
import numpy as np

from fuslib.utils import show_im


def test_cae(data, model, conf):
    print("Testing Convolutionnal AutoEncoder...")
    #Sample outputs
    output = model(data[0:10])
    output = output.detach().numpy()
    
    data = data.numpy()[0:10]
    
    if conf.verbose:
        print(output.shape)
        print(data.shape)

    fig, _ = plt.subplots(nrows=2, ncols=conf.nb_tests, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(conf.nb_tests):
        fig.add_subplot(2, conf.nb_tests, idx+1, xticks=[], yticks=[])
        show_im(data[idx])
    for idx in np.arange(conf.nb_tests):
        fig.add_subplot(2, conf.nb_tests, idx+conf.nb_tests+1, xticks=[], yticks=[])
        show_im(output[idx])
    print("Testing Convolutionnal AutoEncoder done.")

    plt.show()
