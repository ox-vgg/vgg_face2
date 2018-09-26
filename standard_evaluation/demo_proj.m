%% demo script of getting tightly cropped faces using loosely cropped regions

% data_dir is the root directory of your data, e.g. '/vggface2_loose_crop'.
% code_dir is the root directory of your detector code, e.g. '/mtcnn_face',
% including standard caffe, the scripts and dependent libraries supporting MTCNN
% (configuration details can be found in https://github.com/kpzhang93/MTCNN_face_detection_alignment),
% e.g. /mtcnn_face/caffe, /mtcnn_face/mtcnn, /mtcnn_face/pdollar_toolbox.
% det_tight_data_dir is the root directory of your target dir, e.g. '/vggface2_tight_crop'. 
% Please create the subdirectories for classes in advance, e.g. '/vggface2_tight_crop/n000001/'.
% imglist is the name list of the data.

data_dir = YOUR_DATA_DIR;
code_dir = YOUR_CODE_DIR;
det_tight_data_dir = YOUR_TARGET_DIR;
imglist = NAME_LIST;

% path of toolbox
pdollar_toolbox_path= fullfile(code_dir, 'pdollar_toolbox');
code_path = fullfile(code_dir, 'mtcnn/code/codes/MTCNNv2'); %code/codes/MTCNNv2 is the path strcture used in MTCNN's github code. 
addpath(genpath(pdollar_toolbox_path));
addpath(code_path);
caffe_model_path = fullfile(code_path, 'model');
% use cpu
%caffe.set_mode_cpu();
gpu_id = 0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

% three steps's threshold
threshold = [0.6 0.7 0.7];

% scale factor
factor = 0.709;
det_tight_ratio = 0.3;
% load caffe models
prototxt_dir = fullfile(caffe_model_path,'/det1.prototxt');
model_dir = fullfile(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir, model_dir,'test');
prototxt_dir = fullfile(caffe_model_path,'/det2.prototxt');
model_dir = fullfile(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir, model_dir,'test');	
prototxt_dir = fullfile(caffe_model_path,'/det3.prototxt');
model_dir = fullfile(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir, model_dir,'test');
prototxt_dir =  fullfile(caffe_model_path,'/det4.prototxt');
model_dir =  fullfile(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir, model_dir,'test');	

for i = 1: numel(imglist)
    i
    img_name = imglist{i};
    img_path = fullfile(data_dir, img_name);
    img = imread(img_path);
    if size(img,3) == 1
        t_img = repmat(img, 1,1,3);
        img = t_img;
    end
    minl = min([size(img,1) size(img,2)]);
    minsize = fix(minl*0.5);
    tic
    [boundingboxes points] = detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
    toc
    numbox = size(boundingboxes,1);
    scal_ratio_min = 112/128;
    if numbox == 0
        fprintf('%d image cannot match gt\n',i);
    else
        % select the bb closest to the center
        bb_center = [(boundingboxes(:,1) + boundingboxes(:,3))/2, (boundingboxes(:,2) + boundingboxes(:,4))/2];
        bb_center_dist = abs(bb_center - repmat([size(img,2)/2, size(img,1)/2], size(bb_center,1),1));
        [min_dist, region_id] = min(sum(bb_center_dist, 2));
        org_coord = boundingboxes(region_id, 1:2);
        center_point = [(boundingboxes(region_id,1) + boundingboxes(region_id, 3))/2, ...,
        (boundingboxes(region_id,2) + boundingboxes(region_id,4))/2];
        bb_size = boundingboxes(region_id, 3:4) - org_coord;
        scal_ratio = max(scal_ratio_min, bb_size(1)/bb_size(2));
        tight_bb_w_h = bb_size(2) * [scal_ratio, 1] * (1 + det_tight_ratio);
        org_tight_region = img(max(1, round(center_point(2) - tight_bb_w_h(2)/2)): ...
            min(size(img,1), round(center_point(2) + tight_bb_w_h(2)/2)), ...
            max(1, round(center_point(1) - tight_bb_w_h(1)/2)): ...
            min(size(img,2), round(center_point(1) + tight_bb_w_h(1)/2)),:);
        imwrite(org_tight_region, fullfile(det_tight_data_dir, img_name));
    end
end
