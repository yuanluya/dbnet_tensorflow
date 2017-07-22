import numpy as np
import os
import json
import scipy.io as sio
import cv2
import itertools
import operator
import time
import sys
import pdb
from utils import get_scaled_im_tensor, get_scaled_roi,\
                  get_txt_tensor, im2rid, rid2r  
from config import ENV_PATHS, DS_CONFIG
sys.path.append(ENV_PATHS.EDGE_BOX_RPN)

level1_im2p = json.load(open(ENV_PATHS.LEVEL1_TEST, 'r'))
level2_im2p = json.load(open(ENV_PATHS.LEVEL2_TEST, 'r'))

def get_edgeboxes_test(img_id, top_num):
    try:
        raw_boxes = sio.loadmat(os.path.join(ENV_PATHS.EDGEBOX_PATH, 
                                        str(img_id) + '.mat'))['bbs'][0: top_num, :]
    except:
        import edge_boxes
        raw_boxes_ = edge_boxes.get_windows([img_id])[0][0: top_num, :]
        raw_boxes = np.zeros(raw_boxes_.shape)
        raw_boxes[:, 0] = raw_boxes_[:, 1]
        raw_boxes[:, 1] = raw_boxes_[:, 0]
        raw_boxes[:, 2] = raw_boxes_[:, 3] - raw_boxes_[:, 1] + 1
        raw_boxes[:, 3] = raw_boxes_[:, 2] - raw_boxes_[:, 0] + 1

    edge_boxes = np.zeros((0,4))
    edge_boxes = np.concatenate((edge_boxes, raw_boxes[:, 0:4]))
    return edge_boxes

# NMS referenced from http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# for each box, in the format [x1, y1, x2, y2, score]
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # compute the ratio of overlap
        overlap = (w * h) / (area[i] + area[idxs[:last]] - w * h)
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]

def get_pairs_test(img_id, level, edge_box_max, gt_box):
    # (region_id, t_id)
    pair_list = []
    region_ids = im2rid.get(str(img_id))
    if region_ids is None:
        region_ids = []
    edgebox_regions = get_edgeboxes_test(img_id, edge_box_max)
    edgebox_id = "edgebox_%s" % str(img_id)
    region_id = "region_%s" % str(img_id)

    # need to determine how to define the ids
    # currently <source: edgebox|region>_<img_id>_<region_id>
    region_dict = {}
    counter = 0
    for i in range(edgebox_regions.shape[0]):
        e_id = edgebox_id + "_%d" % counter
        region_dict[e_id] = edgebox_regions[i, :]
        counter += 1

    if gt_box:
        phrase_ids = []
        counter = 0
        for rid in region_ids:
            r_info = rid2r[str(rid)]
            r_coord = [r_info['x'], r_info['y'], r_info['width'], r_info['height']]
            r_id = region_id + "_%d" % counter
            region_dict[r_id] = np.array(r_coord)
            # genenrate phrase ids
            phrase_ids.append(r_info['categ_id'])
            counter += 1
    else:
        # genenrate phrase ids
        phrase_ids = [rid2r[str(r)]['categ_id'] for r in region_ids]
    
    if level == 'level_1':
    	phrase_ids = level1_im2p[str(img_id)]
    elif level == 'level_2':
    	phrase_ids = level2_im2p[str(img_id)]
    elif level == 'vis':
        phrase_ids = [-1]
    elif level != 'level_0':
        print('wrong LEVEL parameter, <level_0|1|2>')
        assert(0)

    # generate pair
    pair_list = [(r_id, t_id) for t_id in phrase_ids for r_id in region_dict]

    return pair_list, region_dict

def get_data(img_id, level, edge_box_max, gt_box, query_phrase = None):
    image_tensor, scale, shape = get_scaled_im_tensor([img_id],
                                                      DS_CONFIG.target_size,
                                                      DS_CONFIG.max_size)
    all_rois = np.zeros((0,5))

    # start gathering data for the testing image
    pair_list, region_dict = get_pairs_test(img_id, level, edge_box_max, gt_box)
    rois_list = [pair[0] for pair in pair_list]
    phrases_list = [pair[1] for pair in pair_list]
    unique_rois_ids, inverse_region_ids = (
        np.unique(rois_list, return_inverse = True))
    test_rois = get_scaled_roi(unique_rois_ids, region_dict, 
                               scale[0], shape[0], 0)
    all_rois = np.concatenate((all_rois, test_rois))

    unique_phrase_ids, inverse_phrase_ids = (
        np.unique(phrases_list, return_inverse = True))
    phrase_tensor = get_txt_tensor(unique_phrase_ids, query_phrase)
    
    return (pair_list, region_dict,
           {'raw_phrase': query_phrase,
            'images': image_tensor,
            'phrases': phrase_tensor,
            'rois': all_rois,
            'phrase_ids': inverse_phrase_ids,
            'roi_ids': inverse_region_ids,
            'labels': None, 
            'loss_weights': None,
            'sources': None})

def test_output(img_id, phrase2r_dict, level, output_dir):
    os.makedirs("%s/tmp_output" % output_dir, exist_ok = True)
    f = open("%s/tmp_output/%s_%d.txt" % (output_dir, level, img_id), "w+")
    f.write(str(img_id) + ":")
    for t_id in phrase2r_dict:
        f.write("\n\t%s:" % t_id)
        # output the region informations
        for region in phrase2r_dict[t_id]:
            #write in order [y1, x1, y2, x2]
            f.write(" [%d, %d, %d, %d, %.6f]" %
                (region[1], region[0], region[3], region[2], region[4]))
    f.write("\n")
    f.close()

def test(net, img_id, level, output_dir, top_num = 10, gt_box = False, query_phrase = None):
    if query_phrase is not None:
        assert(level == 'vis')
    t0 = time.time()
    pair_list, region_dict, data_dict = get_data(img_id, level, top_num, gt_box, query_phrase)
    net.set_input(data_dict)
    net.forward(False, False)
    scores = net.get_output()[1]
    scores = [s[0] for s in scores]
    t1 = time.time()
    print ("run through the network takes %f" % (t1 - t0))

    # build region np array for nms
    phrase2r_dict = {}
    combined_region_score = [pair_list[i] + (scores[i],) 
                             for i in range(len(scores))]
    for key, group in itertools.groupby(combined_region_score, 
                                        operator.itemgetter(1)):
        # [x, y, w, h, score]
        regions_info = np.array([np.append(region_dict[info[0]], info[2]) 
                                 for info in list(group)])
        # change from [x, y, w, h] to [x1, y1, x2, y2]
        regions_info[:,2] += regions_info[:,0] - 1
        regions_info[:,3] += regions_info[:,1] - 1

        # apply nms on the top score regions
        regions_info = np.array(
            sorted(regions_info, key = lambda row: row[4])[::-1])
        regions_info_nms = non_max_suppression(regions_info, 0.3)
        phrase2r_dict[key] = regions_info_nms

    t2 = time.time()
    print ("run through the nms takes %f" % (t2 - t1))
    if query_phrase is None:
        test_output(img_id, phrase2r_dict, level, output_dir)
    print ("FINISH TESTING %s" % str(img_id))
    return phrase2r_dict

