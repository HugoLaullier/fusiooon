from torchvision import transforms


def normalize_im(img, conf):
    trans = transforms.Compose([
        transforms.Resize([conf.resize_size, conf.resize_size]),
        transforms.CenterCrop(conf.center_crop_size),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])
    return trans(img)
