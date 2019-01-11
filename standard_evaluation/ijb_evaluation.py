from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np


def get_datalist(s):
    ijbb_meta = np.loadtxt(s, dtype=str)
    faceid = []
    tid = []
    mid = []
    for j in ijbb_meta:
        faceid += [j[0]]
        tid += [int(j[1])]
        mid += [int(j[-1])]

    faceid = np.array(faceid)
    template_id = np.array(tid)
    media_id = np.array(mid)
    return faceid, template_id, media_id


def read_feature_from_bin(path):
    with open(path, 'rb') as fid:
        data_array = np.fromfile(fid, np.float32).reshape((-1, feats_dim))
    return data_array


def template_encoding():
    # ==========================================================
    # 1. face image --> l2 normalization.
    # 2. compute media encoding.
    # 3. compute template encoding.
    # 4. save template features.
    # ==========================================================
    img_feats = read_feature_from_bin(feats_path)  # img_feats --> [number_image x feats_dim]
    img_norm_feats = img_feats / np.sqrt(np.sum(img_feats ** 2, -1, keepdims=True))

    for c, uqt in enumerate(uq_temp):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_norm_feats[ind_t]
        faces_media = medias[ind_t]
        uqm, counts = np.unique(faces_media, return_counts=True)
        media_norm_feats = []

        for u,ct in zip(uqm, counts):
            (ind_m,) = np.where(faces_media == u)
            if ct < 2:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:
                media_norm_feats += [np.sum(face_norm_feats[ind_m], 0, keepdims=True)]

        media_norm_feats = np.array(media_norm_feats)
        media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_norm_feats = np.sum(media_norm_feats, 0)
        np.save(os.path.join(template_path, 'template_{}.npy'.format(uqt)), template_norm_feats)
        if c % 200 == 0: print('Finish Saving {} template features.'.format(c))


def verification():
    # ==========================================================
    #         Loading the Template-specific Features.
    # ==========================================================
    tmp_feats = np.zeros((len(uq_temp), feats_dim))
    for c, uqt in enumerate(uq_temp):
        tmp_feats[c] = np.load(os.path.join(template_path, 'template_{}.npy'.format(uqt)))
        if c % 200 == 0: print('Finish loading {} templates'.format(c))

    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    total_pairs = np.array(range(len(Y)))
    batchsize = 128
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(Y), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        t1 = p1[s]
        t2 = p2[s]
        ind1 = np.squeeze(np.array([np.where(uq_temp == j) for j in t1]))
        ind2 = np.squeeze(np.array([np.where(uq_temp == j) for j in t2]))

        inp1 = tmp_feats[ind1]
        inp2 = tmp_feats[ind2]

        v1 = inp1 / np.sqrt(np.sum(inp1 ** 2, -1, keepdims=True))
        v2 = inp2 / np.sqrt(np.sum(inp2 ** 2, -1, keepdims=True))

        similarity_score = np.sum(v1 * v2, -1)
        score[s] = similarity_score
        if c % 1000 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    np.save(os.path.join(score_path, 'similarity_scores.npy'), score)
    return 1



if __name__ == '__main__':

    print('='*50)
    print('Start processing meta.....')
    print('='*50)

    # =============================================================
    # feats_dim: feature dimension for each image.
    # score_path: similarity score will be saved here.
    # feats_path: features for all images, binary file.
    # template_path: intermediate template encodings will be saved here.
    # =============================================================
    feats_dim = 2048
    score_path = 'scores'
    feats_path = 'feats/IJBB_verify.bin'
    template_path = 'template_features/'

    if not os.path.exists(template_path):
        os.makedirs(template_path)

    if not os.path.exists(score_path):
        os.makedirs(score_path)

    # =============================================================
    # load meta information for template-to-template verification.
    # tid --> template id,  label --> 1/0
    # format:
    #           tid_1 tid_2 label
    # =============================================================
    file = open(os.path.join('meta', 'ijbb_template_pair_label.txt'), 'r')
    meta = file.readlines()
    Y, p1, p2 = [], [], []
    for m in meta:
        msplit = m.split()
        Y += [int(msplit(-1))]
        p1 += [int(msplit(0))]
        p2 += [int(msplit(1))]
    Y, p1, p2 = map(np.array, [Y, p1, p2])
    score = np.zeros((len(p1),))   # cls prediction

    # =============================================================
    # load meta information
    # tid --> template id,  mid --> media id
    # format:
    #           image_name tid mid
    # =============================================================

    faces, templates, medias = get_datalist(os.path.join('meta', 'ijbb_face_tid_mid.txt'))
    uq_temp = np.unique(templates)
    num_uq_temp = len(uq_temp)

    # =============================================================
    # compute template encoding.
    # compute verification.
    # =============================================================
    template_encoding()
    verification()
