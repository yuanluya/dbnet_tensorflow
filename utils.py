"""
Details:
    load the phrase json and track the image
    generate ground truth region & text phrase pair dict
    images need resize
    region id --> (x,y,w,h)
"""
import numpy as np
import os.path as osp
import cv2
import string
import json
import re
from PIL import Image
import pyx
import time
import scipy.io as sio
import scipy.ndimage.interpolation as sni
from collections import defaultdict
from config import DS_CONFIG, ENV_PATHS

#all ids are integer type
t0 = time.clock()
print('<DATA IS LOADING PLEASE BE PATIENT>')
raw_data = json.load(open(ENV_PATHS.RAW_DATA, 'r'))
im2rid = raw_data['im2rid']
rid2r = raw_data['rid2r']
tid2p = raw_data['tid2p']
valid_tids = [int(k) for k in list(tid2p.keys())]

meteor = np.array(json.load(open(ENV_PATHS.METEOR, 'r')))
vocab = [c for c in string.printable if c not in string.ascii_uppercase]
VGG_MEAN = [103.939, 116.779, 123.68]
phrase_freq_temp = json.load(open(ENV_PATHS.FREQUENCY, 'r'))['freq']
phrase_freq = np.zeros(len(phrase_freq_temp))
for _, tid in enumerate(valid_tids):
    phrase_freq[tid - 1] = phrase_freq_temp[tid - 1]
total_frequency = np.sum(phrase_freq)
phrase_prob = phrase_freq / total_frequency

# split
raw_split = json.load(open(ENV_PATHS.SPLIT, 'r'))
train_ids = raw_split['train']
test_ids = raw_split['test']
val_ids = raw_split['val']
print('<DATA LOADING TAKES %f SECONDS>' % (time.clock() - t0))

#[x, y, width, height]
def IoU(region_current, object_current):
    totalarea = (object_current[2] * object_current[3] + 
                 region_current[2] * region_current[3])

    if region_current[0] <= object_current[0]:
        x_left = object_current[0]
    else:
        x_left = region_current[0]

    if region_current[1] <= object_current[1]:
        y_left = object_current[1]
    else:
        y_left = region_current[1]

    if (region_current[0] + region_current[2] >= 
        object_current[0] + object_current[2]):
        x_right = object_current[0] + object_current[2]
    else:
        x_right= region_current[0] + region_current[2]

    if (region_current[1] + region_current[3] >= 
        object_current[1] + object_current[3]):
        y_right = object_current[1] + object_current[3]
    else:
        y_right= region_current[1] + region_current[3]

    if x_right <= x_left:
        intersection = 0
    elif y_right <= y_left:
        intersection = 0
    else:
        intersection = (x_right - x_left) * (y_right - y_left)
    union = totalarea - intersection
    
    return 1.0 * intersection / union

def get_edgeboxes(img_id):
    raw_boxes = sio.loadmat(osp.join(ENV_PATHS.EDGEBOX_PATH, str(img_id) + '.mat'))['bbs']
    chosen_boxes = np.zeros((0, 4))
    chosen_boxes = np.concatenate((chosen_boxes, 
                                   raw_boxes[0: DS_CONFIG.edge_box_high_rank_num, 0:4]))
    rand_ids = (np.random.permutation(
                    raw_boxes.shape[0] - 
                    DS_CONFIG.edge_box_high_rank_num)[0: DS_CONFIG.edge_box_random_num]
                + DS_CONFIG.edge_box_high_rank_num)
    chosen_boxes = np.concatenate((chosen_boxes, raw_boxes[rand_ids, 0:4]))
    return chosen_boxes

def get_label_same_img(img_id):
    #(region_id, t_id): label<0|1|2>
    pair2dict = {}
    ambiguous = {} 
    region_ids = im2rid[str(img_id)]
    edgebox_regions = get_edgeboxes(img_id)

    #get a local dictionary for all regions
    local_region_dict = {}
    counter = 0
    for i in range(edgebox_regions.shape[0]):
        local_region_dict[counter] = edgebox_regions[i, :]
        counter += 1
    for rid in region_ids:
        r_info = rid2r[str(rid)]
        r_coord = [r_info['x'], r_info['y'], r_info['width'], r_info['height']]
        local_region_dict[counter] = np.array(r_coord)
        counter += 1

    #get labels based on IOU
    for r1 in local_region_dict:
        r1_coord = local_region_dict[r1] 
        ambiguous[r1] = []
        for r2 in region_ids:
            t = rid2r[str(r2)]['categ_id']
            r2_info = rid2r[str(r2)]
            r2_coord = [r2_info['x'], r2_info['y'], 
                        r2_info['width'], r2_info['height']]
            iou = IoU(r1_coord, r2_coord)
            if iou <= DS_CONFIG.thre_neg:
                #always get the maximum iou
                if (pair2dict.get((r1, t)) is not None and
                   (pair2dict[(r1, t)] == 1 or pair2dict[(r1, t)] == 2)):
                    continue
                pair2dict[(r1, t)] = 0
            elif iou >= DS_CONFIG.thre_pos:
                if t in ambiguous[r1]:
                    ambiguous[r1].remove(t)
                pair2dict[(r1, t)] = 1
            else:
                #always get the maximum iou
                if (pair2dict.get((r1, t)) is not None and
                    pair2dict[(r1,t)] == 1):
                    continue
                pair2dict[(r1,t)] = 2
                ambiguous[r1].append(t)

    return pair2dict, local_region_dict, ambiguous

def get_label_diff_img(img_id, ambiguous):
    pair2dict = {}
    region_ids = im2rid[str(img_id)]
    #get random sampled phrases out of the given image
    t_ids_in_image = [rid2r[str(r_id)]['categ_id'] for r_id in region_ids]
    rand_t_ids = []
    check_next_tid = 0
    temp_rand = (np.random.choice(len(phrase_prob),
		                 int(1.1 * DS_CONFIG.text_rand_sample_size), 
                                 p = phrase_prob, replace = True) + 1)
    while (len(rand_t_ids) < DS_CONFIG.text_rand_sample_size
           and check_next_tid < len(temp_rand)):
        if temp_rand[check_next_tid] not in t_ids_in_image:
            rand_t_ids.append(temp_rand[check_next_tid])
        check_next_tid += 1
    
    for t_id in rand_t_ids:
        for r_id in ambiguous:
            current_ambiguous = ambiguous[r_id]
            pair2dict[(r_id, t_id)] = 0
            for a_id in current_ambiguous:
                #upper triangle matrix
                if max(t_id, a_id) - 1 >= meteor.shape[0]:
                    continue	
                elif meteor[min(t_id, a_id) - 1, 
                            max(t_id, a_id) - 1] > DS_CONFIG.meteor_thred:
                    pair2dict[(r_id, t_id)] = 2
                    break
    return pair2dict

def get_label_together(img_id):
    pair2dict_same, local_region_dict, ambiguous = get_label_same_img(img_id)
    pair2dict_diff = get_label_diff_img(img_id, ambiguous)
    data_book = []
    #(region_id, categ_id, label, loss_weights, category<pos: 1| neg: 0|rest: 0.5>)
    for b_id, current_p2d in enumerate([pair2dict_same, pair2dict_diff]):
        for k in current_p2d:
            if current_p2d[k] == 1:
                data_book.append([k[0], k[1], 1, DS_CONFIG.pos_loss_weight, 1])
            elif current_p2d[k] == 0 and b_id == 0:#same image negative
                data_book.append([k[0], k[1], 0, DS_CONFIG.neg_loss_weight, 0])
            elif current_p2d[k] == 0 and b_id == 1:#diff image negative
                data_book.append([k[0], k[1], 0, DS_CONFIG.rest_loss_weight, 2])
    return np.array(data_book), local_region_dict

# input a batch of images id
def get_scaled_im_tensor(img_ids, target_size, max_size):
    images = []
    scales = []
    img_shapes = []
    max_w = -1
    max_h = -1
    # load each image
    for img_id in img_ids:
        im_path = osp.join(ENV_PATHS.IMAGE_PATH, str(img_id) + '.jpg')
        try:
            img = cv2.imread(im_path).astype('float')
        except:
            img = cv2.imread(img_id).astype('float')
        img_shapes.append([img.shape[1], img.shape[0]]) #(limit_x, limit_y)
        # calculate scale
        old_short = min(img.shape[0: 2])
        old_long = max(img.shape[0: 2])
        new_scale = 1.0 * target_size / old_short
        if old_long * new_scale > max_size:
            new_scale = 1.0 * max_size / old_long
        # subtract mean from the image
        img[:, :, 0] = img[:, :, 0] - VGG_MEAN[0]
        img[:, :, 1] = img[:, :, 1] - VGG_MEAN[1]
        img[:, :, 2] = img[:, :, 2] - VGG_MEAN[2]
        # scale the image
        img = cv2.resize(img, None, fx = new_scale, fy = new_scale,
                         interpolation = cv2.INTER_LINEAR)
        images.append(img)
        scales.append([new_scale, new_scale])
        # find the max shape
        if img.shape[0] > max_h:
            max_h = img.shape[0]
        if img.shape[1] > max_w:
            max_w = img.shape[1]
    # padding the image to be the max size with 0	
    for idx, img in enumerate(images):
        resize_h = max_h - img.shape[0]
        resize_w = max_w - img.shape[1]
        images[idx] = cv2.copyMakeBorder(img, 0, resize_h, 0, resize_w, 
                                         cv2.BORDER_CONSTANT, value=(0,0,0))

    return np.array(images), np.array(scales), np.array(img_shapes)

def get_txt_tensor(phrase_ids, phrases = None):
    if phrases is None:
        phrases = [tid2p[str(int(id))] for id in phrase_ids]
    else:
        assert(phrase_ids[0] == -1 and len(phrase_ids) == 1)
    tensor = np.zeros([len(phrases), 1, 
                       DS_CONFIG.text_tensor_sequence_length, len(vocab)])
    for idx, line in enumerate(phrases):
        line = line.encode('ascii', errors='ignore')
        line = line.decode('utf-8')
        if line[-1] != '.':
            line = line + ' .'
        line = re.sub(' +', ' ', line)
        line = line.lower()
        line = [char for char in line if char in vocab]
        line = ''.join(line)
        #repeat the phrase to fixed length
        while len(line) < DS_CONFIG.text_tensor_sequence_length:
            line = line + ' ' + line
        for i in range(DS_CONFIG.text_tensor_sequence_length):
            tensor[idx, 0, i, vocab.index(line[i])] = 1
    return tensor

#scale: [x_scale, y_scale]
#shape: [limit_x, limit_y]
#return: [xmin, ymin, xmax, ymax]
#use this to decode local roi dict
def get_scaled_roi(roi_ids, roi_dict, scale, shape, batch_idx, area_thred = 49):
    rois = []
    for idx in roi_ids:
        roi_coor = roi_dict[idx]
        if roi_coor[2] * roi_coor[3] < area_thred:
            continue
        temp_roi = [roi_coor[0] - 1, roi_coor[1] - 1, 
                    roi_coor[0] + roi_coor[2] - 2 , roi_coor[1] + roi_coor[3] - 2]
        rois.append([batch_idx, temp_roi[0] * scale[0], #1-base -> 0-base 
                                temp_roi[1] * scale[1], 
                                temp_roi[2] * scale[0],
                                temp_roi[3] * scale[1]])
    return np.array(rois)

#get all needed data from an image
#image_tensor: [batch_size, width, height, 3]
#phrase_tensor: [num_phrases, 1, sequence_length, vocab_size]
#rois:[batch_idx, xmin, ymin, xmax, ymax]
#pair: [rois_idx, phrase_idx]
#labels|loss_weights: same length as pair
def get_data(img_ids):
    image_tensor, scales, shapes = get_scaled_im_tensor(img_ids, DS_CONFIG.target_size, 
                                                        DS_CONFIG.max_size)
    all_labels = np.zeros((0,))
    all_sources = np.zeros((0,))
    all_loss_weights = np.zeros((0,))
    inverse_region_ids = np.zeros((0,))
    all_rois = np.zeros((0, 5))
    phrases_accumulate = np.zeros((0,))
    #unique roi index is calculated by batch, 
    #when used globally should be offset
    unique_roi_index_offset = 0
    for idx, img_id in enumerate(img_ids):
        current_data_book, current_region_dict = get_label_together(img_id)
        current_unique_rois_ids, current_inverse_ids = ( 
            np.unique(current_data_book[:, 0], return_inverse = True))
        current_rois = get_scaled_roi(current_unique_rois_ids, 
                                      current_region_dict, 
                                      scales[idx, :], shapes[idx,:], idx)
        all_rois = np.concatenate((all_rois, current_rois))
        inverse_region_ids = np.concatenate((inverse_region_ids, 
                current_inverse_ids + unique_roi_index_offset))
        unique_roi_index_offset += current_rois.shape[0]
        #phrase id is unique globally
        phrases_accumulate = np.concatenate((phrases_accumulate, 
                                             current_data_book[:, 1]))	
        all_labels = np.concatenate((all_labels, current_data_book[:, 2]))
        all_loss_weights = np.concatenate((all_loss_weights, 
                                           current_data_book[:, 3]))
        all_sources = np.concatenate((all_sources, current_data_book[:, 4]))
    #get phrase tensor
    unique_phrase_ids, inverse_phrase_ids = np.unique(phrases_accumulate, 
                                                      return_inverse = True)
    phrase_tensor = get_txt_tensor(unique_phrase_ids)

    assert inverse_phrase_ids.shape[0] == inverse_region_ids.shape[0]
    assert inverse_phrase_ids.shape[0] == all_labels.shape[0]
    
    return {'image_ids': img_ids, #for track and debug
            'phrase_ids': unique_phrase_ids, #for track and debug
            'images': image_tensor, 
            'phrases': phrase_tensor, 
            'rois': all_rois, 
            'phrase_ids': inverse_phrase_ids,
            'roi_ids': inverse_region_ids, 
            'labels': all_labels,
            'loss_weights': all_loss_weights,
            'sources': all_sources}

def visualize(im_idx, phrase2ranked, visual_num, phrase, save_path):
    #read in image
    try:
        image = Image.open(osp.join(ENV_PATHS.IMAGE_PATH, str(im_idx) + '.jpg'))
    except:
        image = Image.open(im_idx)
    im_w, im_h = image.size
    ratio = 0.3
    pyx.text.set(mode="latex")
    pyx.text.preamble(r"\renewcommand{\familydefault}{\sfdefault}")
    canv = pyx.canvas.canvas()
    canv.insert(pyx.bitmap.bitmap(0, 0, image, width = ratio * im_w, height = ratio * im_h))
    assert(len(phrase2ranked) == 1)
    ranked = list(phrase2ranked.values())[0]
    for i in range(visual_num):
        (x1, y1, x2, y2, s) = ranked[i]
        w = int(x2 - x1)
        h = int(y2 - y1)
        canv.stroke(pyx.path.rect(ratio * x1, ratio * (im_h - y2), ratio * w, ratio * h),
            [pyx.style.linewidth(1.0), pyx.color.rgb.red])
        #insert score tab for each bbox
        pyx.unit.set(xscale = 3)
        tbox = pyx.text.text(ratio * x1, ratio * (im_h - y1), '[%f]:%s' % (s, phrase), [pyx.text.size.Huge])
        tpath = tbox.bbox().enlarged(3 * pyx.unit.x_pt).path()
        canv.draw(tpath, [pyx.deco.filled([pyx.color.cmyk.Yellow]), pyx.deco.stroked()])
        canv.insert(tbox)

    canv.writePDFfile(save_path)

if __name__ == '__main__':
    main()
