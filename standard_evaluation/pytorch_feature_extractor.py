from __future__ import absolute_import
from __future__ import print_function
import PIL
import torch
import glob as gb
import numpy as np
from PIL import Image


# hyper parameters
batch_size = 10
mean = (131.0912, 103.8827, 91.4953)


def load_data(path='', shape=None):
    short_size = 224.0
    crop_size = shape
    img = PIL.Image.open(path)
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    img = img.convert('RGB')

    ratio = float(short_size) / np.min(im_shape)
    img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
                           int(np.ceil(im_shape[1] * ratio))),  # height
                     resample=PIL.Image.BILINEAR)

    x = np.array(img)  # image has been transposed into (height, width)
    newshape = x.shape[:2]
    h_start = (newshape[0] - crop_size[0])//2
    w_start = (newshape[1] - crop_size[1])//2
    x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    x = x - mean
    return x


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def initialize_model():
    # Download the pytorch model and weights.
    # Currently, it's cpu mode.
    import resnet50_128 as model
    network = model.resnet50_128(weights_path='../model/resnet50_128.pth')
    network.eval()
    return network


def image_encoding(model, facepaths):
    print('==> compute image-level feature encoding.')
    num_faces = len(facepaths)
    face_feats = np.empty((num_faces, 128))
    imgpaths = facepaths
    imgchunks = list(chunks(imgpaths, batch_size))

    for c, imgs in enumerate(imgchunks):
        im_array = np.array([load_data(path=i, shape=(224, 224, 3)) for i in imgs])
        f = model(torch.Tensor(im_array.transpose(0, 3, 1, 2)))[1].detach().cpu().numpy()[:, :, 0, 0]
        start = c * batch_size
        end = min((c + 1) * batch_size, num_faces)
        # This is different from the Keras model where the normalization has been done inside the model.
        face_feats[start:end] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
        if c % 50 == 0:
            print('-> finish encoding {}/{} images.'.format(c * batch_size, num_faces))
    return face_feats


if __name__ == '__main__':
    # rename samples (test set)/tight_crop -> samples
    facepaths = gb.glob('../samples/*/*.jpg')
    model_eval = initialize_model()
    face_feats = image_encoding(model_eval, facepaths)
    S = np.dot(face_feats, face_feats.T)
    import pylab as plt
    plt.imshow(S)
    plt.show()