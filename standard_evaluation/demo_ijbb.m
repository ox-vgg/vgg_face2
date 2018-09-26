%% demo script of getting tightly cropped faces on IJB-B test set.

% data_dir is the root directory of your IJBB data
% code_dir is the root directory of your detector code, e.g. '/mtcnn_face',
% including standard caffe, the scripts and dependent libraries supporting MTCNN
% (configuration details can be found in https://github.com/kpzhang93/MTCNN_face_detection_alignment)
% e.g. /mtcnn_face/caffe, /mtcnn_face/mtcnn, /mtcnn_face/pdollar_toolbox.
% det_tight_data_dir is the root directory for saving crops. Please create the directory "$data_dir/debug" 
% in case of converting image format.

% The coordinates of target facial regions are provided by IJB-B.
% For convenience, we concatenate the files including meta information of faces
% (e.g. ijbb_1N_gallery_S1.csv, ijbb_1N_gallery_S2.csv) and adding the term "FACEID" for face indexing ([1:face_num]).

data_dir = YOUR_IJBB_ROOT_DIR;
code_dir = YOUR_CODE_DIR;
table_path = DATA_CSV_PATH;
det_tight_data_dir = YOUR_TARGET_DIR;

database = readtable(table_path);
imglist = database.FILENAME;
x = database.FACE_X;
y = database.FACE_Y;
width = database.FACE_WIDTH;
height = database.FACE_HEIGHT;
faceid = database.FACEID; %index the target faces
%path of toolbox
pdollar_toolbox_path = fullfile(code_dir, 'pdollar_toolbox';
code_path = fullfile(code_dir, 'mtcnn/code/codes/MTCNNv2');%code/codes/MTCNNv2 is the path strcture used in MTCNN's github code. 
addpath(genpath(pdollar_toolbox_path));
addpath(code_path);
caffe_model_path = fullfile(code_path, '/model');
%use cpu
%caffe.set_mode_cpu();
gpu_id = 0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7];
%scale factor
factor=0.709;
det_tight_ratio = 0.3;
%load caffe models
prototxt_dir =fullfile(caffe_model_path,'/det1.prototxt');
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

% images detector failed on
non_match_gt = zeros(length(imglist),1);
for i = 1: length(imglist)
    i
    img_name = imglist{i};
    img_path = fullfile(data_dir, img_name);
    try 
 	img = imread(img_path);
    catch
        f1 = fullfile(data_dir, ['debug/', img_name]);
        f2 = fullfile(data_dir, ['debug/1_', img_name]);
        copyfile(img_path, f1);
        system(sprintf('convert -colorspace RGB %s %s', f1, f2));
        img = imread(f2);
    end
    gt_box = [x(i), y(i), x(i) + width(i), y(i) + height(i)];
    % crop a loose candidate region
    center_gt = [x(i) + 0.5 * width(i), y(i) + 0.5 * height(i)];
    extent_gt = 1.4;
    candidate_region = img(max(1, round(center_gt(2) - height(i)* extent_gt /2)): min(size(img,1), ...
        round(center_gt(2) + height(i) * extent_gt /2)), max(1, round(center_gt(1) - width(i) * extent_gt /2)): ...
        min(size(img,2), round(center_gt(1) + width(i) * extent_gt /2)),:);
    img = candidate_region;
    %we recommend you to set minsize as x * short side
    if size(img,3) == 1
        t_img = repmat(img, 1,1,3);
        img = t_img;
    end
    minl = min([size(img,1) size(img,2)]);
    minsize = fix(minl*0.1);
    tic
    [boundingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
    toc
    numbox=size(boundingboxes,1);
    scal_ratio_min = 112/128;
    if numbox == 0
        non_match_gt(i) = 1;
        fprintf('%d image cannot match gt\n',i);
    else
      % select the bb close to the center
      bb_center = [(boundingboxes(:,1) + boundingboxes(:,3))/2, (boundingboxes(:,2) + boundingboxes(:,4))/2];
      bb_gt_dist = abs(bb_center - repmat([size(img,2)/2, size(img,1)/2], size(bb_center,1),1));
      [min_dist, region_id]= min(sum(bb_gt_dist, 2));
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
      imwrite(org_tight_region, fullfile(det_tight_data_dir, [num2str(faceid(i)), '.jpg']));
    end
end

% dealing with empty files
extend_ratio = 0.1;
non_ind = find(non_match_gt);
for i = 1:numel(non_ind)
    i
    img_name = imglist{non_ind(i)};
    img_path = fullfile(data_dir, img_name);
    img = imread(img_path);
    %cropping with GT_0.1
    bb_width = width(non_ind(i)) * (1 + extend_ratio);
    bb_height = height(non_ind(i)) * (1 + extend_ratio);
    xy = [x(non_ind(i)), y(non_ind(i))];
    centrepoint = [round(xy(1) + bb_width/2) round(xy(2) + bb_height/2)]; 
    x1 = centrepoint(1) - round(bb_width / 2);
    y1 = centrepoint(2) - round(bb_height / 2);
    x2 = centrepoint(1) + round(bb_width / 2);
    y2 = centrepoint(2) + round(bb_height / 2);
    
    x1 = max(1, x1);
    y1 = max(1, y1);
    x2 = min(x2, size(img,2));
    y2 = min(y2, size(img,1));
    
    crop = img(y1:y2, x1:x2, :);
    imwrite(crop, fullfile(det_tight_data_dir, [num2str(faceid(non_ind(i))), '.jpg']));
end
