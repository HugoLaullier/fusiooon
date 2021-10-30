from fuslib import modules, opts

if __name__ == '__main__':
    modules.struct(conf=opts.Struct())
    imgs = modules.get_data(conf=opts.GetData())
    dl = modules.preproc(imgs, conf=opts.Preproc())
    model = modules.train_cae(dl, conf=opts.TrainCAE())
    modules.test_cae(imgs, model, conf=opts.TestCAE())
    modules.trans_cae(model, conf=opts.TransCAE())


