# modified from https://github.com/JonathonLuiten/TrackEval
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment


def caculate_similarity(gts, preds, do_ioa=False):
    # caculate similarity between gt and pred
    # gts: [track_id, l, t, r, b, score, class]
    # preds: [track_id, l, t, r, b, score, class]
    # return: similarity matrix
    maxs = np.maximum(gts[:, np.newaxis, :], preds[np.newaxis, :, :])
    mins = np.minimum(gts[:, np.newaxis, :], preds[np.newaxis, :, :])
    intersects = np.maximum(
        0, mins[:, :, 3] - maxs[:, :, 1]) * np.maximum(0, mins[:, :, 4] - maxs[:, :, 2])
    bbox1_areas = (gts[:, 3] - gts[:, 1]) * (gts[:, 4] - gts[:, 2])
    if do_ioa:
        ioas = np.zeros_like(intersects)
        mask = bbox1_areas > 0
        ioas[mask, :] = intersects[mask, :] / bbox1_areas[mask, np.newaxis]

    bbox2_areas = (preds[:, 3] - preds[:, 1]) * (preds[:, 4] - preds[:, 2])
    union = bbox1_areas[:, np.newaxis] + \
        bbox2_areas[np.newaxis, :] - intersects
    # following three line not sure why, copied from trackeval
    intersects[bbox1_areas < np.finfo(float).eps, :] = 0
    intersects[:, bbox2_areas < np.finfo(float).eps] = 0
    intersects[union < np.finfo(float).eps] = 0
    union[union < np.finfo(float).eps] = 1
    ious = intersects / union
    return ious


def hota_maximize_solve(ground_truth, predictions):
    # ground_truth: [frame_id[track_id, l, t, r, b, score, class]]
    # predictions: [frame_id[track_id, l, t, r, b, score, class]]
    # iou_threshold: iou threshold for matching
    # match_pair: [threshhold[frame_id[match_rows, match_cols]]]
    #             the rows and cols are indexes of original gt and pred

    gt_unique_ids = np.sort(
        np.unique([int(item) for list in ground_truth for item in list[:, 0]]))
    pred_unique_ids = np.sort(
        np.unique([int(item) for list in predictions for item in list[:, 0]]))
    gt_num_ids = len(gt_unique_ids)
    pred_num_ids = len(pred_unique_ids)
    gt_id_map = {id: i for i, id in enumerate(gt_unique_ids)}
    pred_id_map = {id: i for i, id in enumerate(pred_unique_ids)}
    potential_match_count = np.zeros((gt_num_ids, pred_num_ids))
    similarities = []
    gt_ids, pred_ids = [], []
    gt_id_count = np.zeros(gt_num_ids)
    pred_id_count = np.zeros(pred_num_ids)
    thresh_array = np.arange(0.05, 1, 0.05)
    match_pair = [[] for _ in thresh_array]

    for t, gt, pred in zip(range(len(ground_truth)), ground_truth, predictions):
        similarities.append(caculate_similarity(gt, pred))
        gt_ids.append([int(item) for item in gt[:, 0]])
        pred_ids.append([int(item) for item in pred[:, 0]])
        similarity = similarities[-1]
        sim_denom = np.sum(similarity, axis=0)[
            np.newaxis, :] + np.sum(similarity, axis=1)[:, np.newaxis] - similarity
        mask = sim_denom > np.finfo(float).eps
        sim_iou = np.zeros_like(similarity)
        sim_iou[mask] = similarity[mask] / sim_denom[mask]
        gt_indexes = np.array([gt_id_map[id] for id in gt_ids[-1]])
        pred_indexes = np.array([pred_id_map[id] for id in pred_ids[-1]])
        potential_match_count[gt_indexes[:, np.newaxis],
                              pred_indexes[np.newaxis, :]] += sim_iou
        gt_id_count[gt_indexes] += 1
        pred_id_count[pred_indexes] += 1

    global_alignment_scores = potential_match_count / \
        (gt_id_count[:, np.newaxis] +
         pred_id_count[np.newaxis, :] - potential_match_count)

    for t, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        similarity = similarities[t]
        gt_indexes = np.array([gt_id_map[id] for id in gt[:, 0]])
        pred_indexes = np.array([pred_id_map[id] for id in pred[:, 0]])

        sim_score = global_alignment_scores[gt_indexes[:,
                                                       np.newaxis], pred_indexes[np.newaxis, :]] * similarity
        match_rows, match_cols = linear_sum_assignment(-sim_score)
        for a, alpha in enumerate(thresh_array):
            mask = similarity[match_rows,
                              match_cols] >= alpha - np.finfo(float).eps
            match_pair[a].append(
                [gt[:, 0][match_rows[mask]], pred[:, 0][match_cols[mask]]])

    return match_pair


def hota_eval(gt_ids, pred_ids, match_pair):
    thresh_array = np.arange(0.05, 1, 0.05)
    gt_ids_global = np.sort(
        np.unique([item for list in gt_ids for item in list]))
    pred_id_global = np.sort(
        np.unique([item for list in pred_ids for item in list]))
    gt_id_map = {id: i for i, id in enumerate(gt_ids_global)}
    pred_id_map = {id: i for i, id in enumerate(pred_id_global)}
    match_counts = [np.zeros((len(gt_ids_global), len(pred_id_global)))
                    for _ in thresh_array]
    gt_id_count = np.zeros(len(gt_ids_global))
    pred_id_count = np.zeros(len(pred_id_global))
    res = {}
    res['HOTA_TP'] = []
    for a, alpha in enumerate(thresh_array):
        gt_id_count.fill(0)
        pred_id_count.fill(0)
        for t, (_, _) in enumerate(zip(gt_ids, pred_ids)):
            match_rows, match_columns = match_pair[a][t]
            match_count = match_counts[a]
            gt_indexes = [gt_id_map[id] for id in match_rows]
            pred_indexes = [pred_id_map[id] for id in match_columns]
            match_count[gt_indexes, pred_indexes] += 1
            gt_id_count[[gt_id_map[id] for id in gt_ids[t]]] += 1
            pred_id_count[[pred_id_map[id] for id in pred_ids[t]]] += 1

        ass_a = match_count / \
            np.maximum(
                1, (gt_id_count[:, np.newaxis] + pred_id_count[np.newaxis, :] - match_count))
        res['HOTA_TP'].append(
            np.sum(match_count * ass_a) / np.sum(match_count))
        print(alpha, res['HOTA_TP'][a])

def preproc(ground_truth, predictions):
    for t, _, _ in zip(range(len(ground_truth)), ground_truth, predictions):
        ground_truth[t][:, 3] = ground_truth[t][:, 1] + ground_truth[t][:, 3]
        ground_truth[t][:, 4] = ground_truth[t][:, 2] + ground_truth[t][:, 4]
        predictions[t][:, 3] = predictions[t][:, 1] + predictions[t][:, 3]
        predictions[t][:, 4] = predictions[t][:, 2] + predictions[t][:, 4]
        similarity = caculate_similarity(ground_truth[t], predictions[t])
        similarity[similarity < 0.5 - np.finfo('float').eps] = 0
        match_rows, match_columns = linear_sum_assignment(-similarity)
        mask = similarity[match_rows, match_columns] >= np.finfo('float').eps
        match_rows = match_rows[mask]
        match_columns = match_columns[mask]
        distract = np.isin( ground_truth[t][match_rows][:, 6], [2, 6, 7, 8, 12])
        to_remove_tracker = match_columns[distract]
        predictions[t] = np.delete(predictions[t], to_remove_tracker, axis=0)
        zero_masked = ground_truth[t][:, 5] == 0
        ground_truth[t] = np.delete(ground_truth[t], zero_masked, axis=0)
    return ground_truth, predictions

if __name__ == '__main__':
    gt_file = '/home/ubuntu/repos/mot/trackeval/data/gt/mot_challenge/MOT20-train/MOT20-01/gt/gt.txt'
    pred_file = '/home/ubuntu/repos/mot/trackeval/data/trackers/mot_challenge/MOT20-train/xinCOLOR/data/MOT20-01.txt'
    gts = np.loadtxt(gt_file, delimiter=',')
    time_ids = np.sort(np.unique(gts[:, 0]))
    ground_truth = [gts[gts[:, 0] == time_id][:, 1:] for time_id in time_ids]
    preds = np.loadtxt(pred_file, delimiter=',')
    predictions = [preds[preds[:, 0] == time_id][:, 1:9]
                   for time_id in time_ids]
    ground_truth, predictions = preproc(ground_truth, predictions)
    matched_pairs = hota_maximize_solve(ground_truth, predictions)
    gt_ids = [list[:, 0] for list in ground_truth]
    pred_ids = [list[:, 0] for list in predictions]
    hota_eval(gt_ids, pred_ids, matched_pairs)
