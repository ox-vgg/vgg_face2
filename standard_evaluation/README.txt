This folder contains the example code for cropping on IJBB (and VGGFace2) and evaluation code for IJBB dataset.

It will use the following information

"meta/": contain the meta informations, ijbb_face_tid_mid.txt, ijbb_template_pair_label.txt.

"feats": please compute image features and save them as a single binary file.
check the ijbb_face_tid_mid.txt to see the order for image features (faces are indexed from 1 to n based on the order of protocols {ijbb_1N_gallery_S1, ijbb_1N_gallery_S2, ijbb_1N_probe_mixed}. 

"scores/": output similarity scores will be saved here. 

