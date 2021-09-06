import os

ImageSets_root = './ImageSets'
Main_root = '/Main'
if not os.path.exists(ImageSets_root):
    os.makedirs(ImageSets_root)
    os.makedirs(ImageSets_root + Main_root)

if __name__ == '__main__':
    txt_name = '/all.txt'
    txt_path = ImageSets_root + Main_root + txt_name
    image_path = './JPEGImages'
    ids = list()
    l = os.listdir(image_path)

    ids = [x.split('.')[0] for x in l]

    with open(txt_path, "w") as txt_f:
        for imgId in ids:
            print(imgId, file=txt_f)
