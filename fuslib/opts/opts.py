import torch.nn as nn

class BaseOpts():
    def __init__(self):
        self.work_dir = "/home/hugo/Documents/Fusiooon/fusiooon/structure"
        self.data_dir_train = "/home/hugo/Documents/Fusiooon/cats/"
        self.data_dir_test = "/home/hugo/Documents/Fusiooon/teckel/"

class NormalizeIm():
    def __init__(self):
        self.resize_size = 256
        self.center_crop_size = 224

class BaseModel():
    def __init__(self):
        self.batch_size = 32

class Struct(BaseOpts):
    def __init__(self):
        BaseOpts.__init__(self)
        self.__name__ = "Struct"
        self.remove_struct = False
        self.verbose = False

class GetData(BaseOpts, NormalizeIm):
    def __init__(self):
        BaseOpts.__init__(self)
        NormalizeIm.__init__(self)
        self.__name__ = "GetData"
        self.store = False
        self.nb_images = 1500
        self.verbose = True

class Preproc(BaseOpts):
    def __init__(self):
        BaseOpts.__init__(self)
        self.__name__ = "Preproc"
        self.batch_size = 32
        self.verbose = False

class TrainCAE(BaseOpts):
    def __init__(self):
        BaseOpts.__init__(self)
        BaseModel.__init__(self)
        self.__name__ = "TrainCAE"
        self.train = True
        self.model = "CAE"
        self.nb_epochs = 100
        self.criterion = nn.BCELoss()
        self.optim = 'Adam'
        self.lr = 0.001
        self.verbose = True

class TestCAE(BaseOpts):
    def __init__(self):
        BaseOpts.__init__(self)
        BaseModel.__init__(self)
        self.__name__ = "TestCAE"
        self.nb_tests = 5
        self.verbose = False

class TransCAE(BaseOpts):
    def __init__(self):
        BaseOpts.__init__(self)
        BaseModel.__init__(self)
        NormalizeIm.__init__(self)
        self.__name__ = "TransCAE"
        self.verbose = False
