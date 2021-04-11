import numpy as np
import torch


def generate_anchors(stride, ratio_vals, scales_vals, angles_vals=None):
    'Generate anchors coordinates from scales/ratios'
    #print("scales_vals", scales_vals)
    scales = np.tile(scales_vals, [len(ratio_vals), 1]).astype(np.float32)
    scales = scales.transpose(1, 0).reshape(-1, 1).astype(np.float32)

    ratios = ratio_vals * len(scales_vals)

    wh = np.tile([stride], [len(ratios), 2]).astype(np.float32)

    ws = np.sqrt(wh[:, 0] * wh[:, 1] / np.array(ratios).astype(np.float32)).astype(np.float32)
    
    dwh = np.stack((ws, ws * ratios), axis=1).astype(np.float32)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return np.concatenate((xy1, xy2), axis=1).astype(np.float32)


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    'Non Maximum Suppression'
    from test_2 import compare_result

    batch_size = all_scores.shape[0]
    out_scores = np.zeros((batch_size, ndetections))
    out_boxes = np.zeros((batch_size, ndetections, 4))
    out_classes = np.zeros((batch_size, ndetections))

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = np.transpose(np.asarray((all_scores[batch, :].reshape(-1) > 0).nonzero()))
        scores = all_scores[batch, keep].reshape(-1)
        boxes = all_boxes[batch, keep, :].reshape(-1, 4)
        classes = all_classes[batch, keep].reshape(-1)

        if scores.size == 0:
            continue

        # Sort boxes
        indices = np.argsort(scores)[::-1]
        scores = scores[indices]
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).reshape(-1)
        keep = np.ones(scores.shape, dtype=np.uint8).reshape(-1)

        for i in range(ndetections):
            # if i >= keep.nonzero().nelement() or i >= scores.nelement():
            if i >= np.asarray(keep.nonzero()).size or i >= scores.size:
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = np.maximum(boxes[:, :2], boxes[i, :2])
            xy2 = np.minimum(boxes[:, 2:], boxes[i, 2:])
            inter = np.prod((xy2 - xy1 + 1).clip(0), 1)

            boxes_std = torch.from_numpy(boxes)
            xy1_std = torch.max(boxes_std[:, :2], boxes_std[i, :2])
            xy2_std = torch.min(boxes_std[:, 2:], boxes_std[i, 2:])
            inter_std = torch.prod((xy2_std - xy1_std + 1).clamp(0), 1)
            areas_std = torch.from_numpy(areas)
            classes_std = torch.from_numpy(classes)
            scores_std = torch.from_numpy(scores)

            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1

            criterion_std = ((scores_std > scores_std[i]) |
                         (inter_std / (areas_std + areas_std[i] - inter_std) <= nms) |
                         (classes_std != classes_std[i]))
            criterion_std[i] = 1
            compare_result(criterion, criterion_std)

            # Only keep relevant boxes
            scores = scores[np.transpose(np.asarray(criterion.nonzero()))].reshape(-1)
            boxes = boxes[np.transpose(np.asarray(criterion.nonzero())), :].reshape(-1, 4)
            classes = classes[np.transpose(np.asarray(criterion.nonzero()))].reshape(-1)
            areas = areas[np.transpose(np.asarray(criterion.nonzero()))].reshape(-1)
            keep[np.transpose(np.asarray((~criterion).nonzero()))] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes


def delta2box1(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = np.exp(deltas[:, 2:]) * anchors_wh

    m = np.zeros([2], dtype=deltas.dtype)
    M = (np.array([size], dtype=deltas.dtype) * stride - 1)
    clamp = lambda t: np.maximum(m, np.minimum(t, M))
    return np.concatenate((
        clamp(pred_ctr - 0.5 * pred_wh),
        clamp(pred_ctr + 0.5 * pred_wh - 1)
    ), 1)


def delta2box(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = (torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1)
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat([
        clamp(pred_ctr - 0.5 * pred_wh),
        clamp(pred_ctr + 0.5 * pred_wh - 1)
    ], 1)


def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None, rotated=False):
    'Box Decoding and Filtering'
    from test_2 import compare_result

    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6

    device = "cpu"
    anchors = anchors.astype(all_cls_head.dtype)
    num_anchors = anchors.shape[0] if anchors is not None else 1
    num_classes = all_cls_head.shape[1] // num_anchors
    height, width = all_cls_head.shape[-2:]
    batch_size = all_cls_head.shape[0]
    out_scores = np.zeros((batch_size, top_n))
    out_boxes = np.zeros((batch_size, top_n, num_boxes))
    out_classes = np.zeros((batch_size, top_n))
    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].reshape(-1)
        box_head = all_box_head[batch, :, :, :].reshape(-1, num_boxes)
        keep = np.asarray((cls_head >= threshold).nonzero()).transpose().reshape(-1)
        if keep.size == 0:
            continue
        # Gather top elements
        scores = np.take(cls_head, keep, 0)
        scores, indices = np_topk_(scores, min(top_n, keep.shape[0]), 0)
        indices = np.take(keep, indices, 0).reshape(-1)
        print(indices)
        #classes = ((indices.astype(np.float32) / height) / width)% num_classes
        classes = np.floor_divide(np.floor_divide(indices, width), height) % num_classes
        classes = classes.astype(all_cls_head.dtype)
        # Infer kept bboxes
        x = indices % width
        y = np.floor_divide(indices, width) % height
        a = np.floor_divide(np.floor_divide(np.floor_divide(indices, num_classes), height), width)
        # a = (((indices / num_classes) / height) / width)

        box_head = box_head.reshape(num_anchors, num_boxes, height, width)
  
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = np.stack((x, y, x, y), 1).astype(all_cls_head.dtype) * stride + anchors[a, :]
            boxes = delta2box(torch.from_numpy(boxes), torch.from_numpy(grid), [width, height], stride).numpy()
        out_scores[batch, :scores.shape[0]] = scores
        out_boxes[batch, :boxes.shape[0], :] = boxes
        out_classes[batch, :classes.shape[0]] = classes

    return out_scores, out_boxes, out_classes



def np_topk_(matrix, K, axis=1):
    if axis == 0:
        if len(matrix.shape) == 1:
            row_index = np.arange(matrix.shape[0])
            if K == matrix.shape[0]:
                topk_index = np.argpartition(-matrix, K-1, axis=axis)[0:K]
            else:
                topk_index = np.argpartition(-matrix, K, axis=axis)[0:K]
            topk_data = matrix[topk_index]
            topk_index_sort = np.argsort(-topk_data,axis=axis)
            topk_data_sort = topk_data[topk_index_sort]
            topk_index_sort = topk_index[0:K][topk_index_sort]
        else:
            row_index = np.arange(matrix.shape[1 - axis])
            topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
            topk_data = matrix[topk_index, row_index]
            topk_index_sort = np.argsort(-topk_data,axis=axis)
            topk_data_sort = topk_data[topk_index_sort,row_index]
            topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort
