"""

@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}

"""

import os
import torch
import matplotlib.pyplot as plt
from CaffeModel.colorizers import *
from CaffeParameter import get_parameters


def getPictureName(path):
    return os.path.splitext(os.path.basename(path))[0]


def saveColorizedPicture(image, directory, filename):
    path = os.path.join(directory, filename)
    plt.imsave(path, image)
    #print(f"Saved: {path}")


def main():
    opt = get_parameters()

    pictureName = getPictureName(opt.img_path)

    eccv16Directory = os.path.join(opt.output_dir, 'ECCV16')
    siggraph17Directory = os.path.join(opt.output_dir, 'SIGGRAPH17')
    os.makedirs(eccv16Directory, exist_ok=True)
    os.makedirs(siggraph17Directory, exist_ok=True)

    if opt.model in ['eccv16', 'both']:
        eccv16Colorizer = eccv16(pretrained=True).eval()
        if opt.use_gpu:
            eccv16Colorizer.cuda()

    if opt.model in ['siggraph17', 'both']:
        siggraph17Colorizer = siggraph17(pretrained=True).eval()
        if opt.use_gpu:
            siggraph17Colorizer.cuda()

    img = load_img(opt.img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    if opt.use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    BWPicture = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))

    # Process and save picture with selected models
    if opt.model in ['eccv16', 'both']:
        out_img_eccv16 = postprocess_tens(tens_l_orig, eccv16Colorizer(tens_l_rs).cpu())
        saveColorizedPicture(out_img_eccv16, eccv16Directory, f"{pictureName}_ECCV16.png")

    if opt.model in ['siggraph17', 'both']:
        out_img_siggraph17 = postprocess_tens(tens_l_orig, siggraph17Colorizer(tens_l_rs).cpu())
        saveColorizedPicture(out_img_siggraph17, siggraph17Directory, f"{pictureName}_SIGGRAPH17.png")

    # Show results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(BWPicture)
    plt.title('Input (Black & White)')
    plt.axis('off')

    if opt.model in ['eccv16', 'both']:
        plt.subplot(2, 2, 3)
        plt.imshow(out_img_eccv16)
        plt.title('Output (ECCV 16)')
        plt.axis('off')

    if opt.model in ['siggraph17', 'both']:
        plt.subplot(2, 2, 4)
        plt.imshow(out_img_siggraph17)
        plt.title('Output (SIGGRAPH 17)')
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
