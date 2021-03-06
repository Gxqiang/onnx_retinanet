#import numpy as np 
import onnxruntime
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from onnx import numpy_helper




def ort_inference(file, inputs_flatten, result):
    print("====== ORT Inference ======")
    ort_sess = onnxruntime.InferenceSession(file)
    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs_flatten))
    ort_outs = ort_sess.run(None, ort_inputs)
    result.append(ort_outs)


def preprocess(img, means = [0.485, 0.456, 0.406], stds = [0.229, 0.224, 0.225]):
    img = np.transpose(img, (2, 0, 1))
    for t, mean, std in zip(img, means, stds):
        t = (t / 255  + mean) / std
    img = img.astype(np.float32)
    
    return img[np.newaxis, :, :, :]


def generate_anchors(stride, ratio_vals, scales_vals, angles_vals=None):
    'Generate anchors coordinates from scales/ratios'

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return torch.cat([xy1, xy2], dim=1)


def box2delta(boxes, anchors):
    'Convert boxes to deltas from anchors'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    return torch.cat([
        (boxes_ctr - anchors_ctr) / anchors_wh,
        torch.log(boxes_wh / anchors_wh)
    ], 1)


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

    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6

    device = "cpu"
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, num_boxes), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, num_boxes)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        #classes = (indices / width / height) % num_classes
        classes = torch.floor_divide(torch.floor_divide(indices, width), height) % num_classes
        classes = classes.type(all_cls_head.type())
        print(indices)

        # Infer kept bboxes
        x = indices % width
        #y = (indices / width).type(torch.LongTensor) % height
        y = torch.floor_divide(indices, width) % height
        # a = (((indices / num_classes).type(torch.LongTensor) / height).type(torch.LongTensor) / width).type(torch.LongTensor)
        a = torch.floor_divide(torch.floor_divide(torch.floor_divide(indices, num_classes), height), width)
        box_head = box_head.view(num_anchors, num_boxes, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride + anchors[a, :]
            boxes = delta2box(boxes, grid, [width, height], stride)
        out_scores[batch, :scores.size()[0]] = scores
        out_boxes[batch, :boxes.size()[0], :] = boxes
        out_classes[batch, :classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    'Non Maximum Suppression'

    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 4), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)


        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)

        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 1)

            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes

def detection_postprocess(image, cls_heads, box_heads):
    anchors = {}
    decoded = []

    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        stride = image.shape[-2] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
        # Decode and filter boxes
        print(stride)
        decoded.append(decode(cls_head, box_head, stride,
                              threshold=0.05, top_n=1000, anchors=anchors[stride]))

    # Perform non-maximum suppression
    decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
    # NMS threshold = 0.5
    scores, boxes, labels = nms(*decoded, nms=0.5, ndetections=1000)
    return scores, boxes, labels

def detection_postprocess_2(image, cls_heads, box_heads):
    # Inference post-processing
    import box
    anchors = {}
    anchors_std = {}
    decoded = []
    decoded_std = []
    np.set_printoptions(4,threshold=np.inf)
    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        stride = image.shape[-2] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = box.generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
            
            anchors_std[stride] = generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
            #err = np.sum(np.abs(anchors[stride] - anchors_std[stride].numpy()))
            #print(err)
        # Decode and filter boxes
        print(stride)
        compare_result(anchors[stride], anchors_std[stride])
        decoded.append(box.decode(cls_head.numpy().astype(np.float32), box_head.numpy().astype(np.float32), stride,
                              threshold=0.05, top_n=1000, anchors=anchors_std[stride].numpy().astype(np.float32)))
                              #threshold=0.05, top_n=1000, anchors=anchors[stride].astype(np.float32)))
        decoded_std.append(decode(cls_head, box_head, stride,
                              threshold=0.05, top_n=1000, anchors=anchors_std[stride]))
    
    # Perform non-maximum suppression
    print("=======compare decode =====")
    
    compare_result(decoded, decoded_std)
    decoded = [np.concatenate(tensors, 1) for tensors in zip(*decoded)]
    #decoded = [np.concatenate((tensors[0].numpy(), tensors[1].numpy(),tensors[2].numpy(),tensors[3].numpy(),tensors[4].numpy()), 1) for tensors in zip(*decoded_std)]
    decoded_std = [torch.cat(tensors, 1) for tensors in zip(*decoded_std)]
    print("-----")
    compare_result(decoded, decoded_std)
    
    #print(decoded)
    #print(decoded_std)
    print()

    # NMS threshold = 0.5
    scores, boxes, labels = box.nms(*decoded, nms=0.5, ndetections=100)
    scores_std, boxes_std, labels_std = nms(*decoded_std, nms=0.5, ndetections=100)
    print("=======compare nms=====")
    compare_result(scores, scores_std)
    compare_result(boxes, boxes_std)
    compare_result(labels, labels_std)
    print("============")
    return scores, boxes, labels

def compare_result(a,b):
    N = len(a)
    for i in range(N):
        if isinstance(a[i], tuple):
            for j in range(len(a[i])):
                err = np.sum(np.abs(a[i][j].astype(np.float64) - b[i][j].numpy().astype(np.float64)))
                if err > 0.0:
                    print(err)
        else:
            err = np.sum(np.abs(a[i].astype(np.float64) - b[i].numpy().astype(np.float64)))
            if err > 0.0:
                print(err)
            #if err > 1:
                #print(a[i]- b[i].numpy())


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list
from torchvision import transforms
import torch.nn.functional as F

def resize_im(img, output_shape = [480, 640]):
    h, w = img.size
    r_h = output_shape[1] / h
    r_w = output_shape[0] / w
    r = max(int(min([r_h,r_w])), 1)
    h_resize = r * h
    w_resize = r * w
    img = img.resize((h_resize, w_resize),Image.ANTIALIAS)
    return img, r

def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


filename = "D:\Programs\work_dir\\000000046252.jpg" #demo.jpg"
#filename = "D:\Programs\work_dir\\val2017\\000000135604.jpg" 
#filename = "D:\Programs\work_dir\\val2017\\000000397133.jpg"
#filenames = ["D:\Programs\work_dir\\000000046252.jpg"]#, 
            # "D:\Programs\work_dir\\val2017\\000000135604.jpg",
            # "D:\Programs\work_dir\\val2017\\000000397133.jpg"]
model_dir = 'D:\Programs\work_dir\\test_retinanet_resnet101\\retinanet-9.onnx'
json_path = "D:\Programs\work_dir\\instances_val2017.json"
results = []
# image_ids = [46252] #[46252, 135604, 397133] #[135604] #[46252]
import os
import onnx

path = "D:\Programs\work_dir\\val2017\\"
#for filename,image_id in zip(filenames,image_ids):
for file in os.listdir(path):
    filename = os.path.join(path, file) 
    image_id = int(file.split('.')[0])
    input_image = Image.open(filename).convert("RGB")
    input_image,ratios  = resize_im(input_image)
    in_w, in_h = input_image.size
    print("image size???", input_image.size)
    input_image = np.array(input_image)
    preprocess1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess1(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    print("input_tesnor:", input_tensor.size()) 

    pw = 640 - in_w
    ph = 480 - in_h
    input_tensor = F.pad(input_tensor, (0, pw, 0, ph))
    # Test exported model with TensorProto data saved in files
    inputs_flatten = flatten(input_tensor.detach().cpu().numpy())
    print(inputs_flatten[0].shape)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    print(inputs_flatten[0].shape)
    

    ids = [image_id]#[397133] #[135604] #[46252]

    #img = preprocess(input_image)
    outputs = []
    ort_inference(model_dir, inputs_flatten, outputs)
    cls_heads = []
    box_heads = []
    for i in range(len(outputs[0])):
        if i < 5:
            cls_heads.append(torch.from_numpy(outputs[0][i]))
        else:
            box_heads.append(torch.from_numpy(outputs[0][i]))

#print("cls_heads:", cls_heads.shape)
#print("box_heads:", box_heads)
    scores, boxes, classes = detection_postprocess_2(input_image, cls_heads, box_heads)
    #scores, boxes, classes = detection_postprocess(input_image, cls_heads, box_heads)
    #print("scores:", np.sum(np.abs(np.array(scores) - scores2.numpy()))) 
    #print("boxes:", np.sum(np.abs(np.array(boxes) - boxes2.numpy() )))
    #print("labels:", np.sum(np.abs(np.array(classes) - classes2.numpy()) ))
    results.append([scores, boxes, classes, ids, [ratios]])
    if len(results) > 3:
        break
#results = []
detections = []
processed_ids = [] #set()
coco = COCO(json_path) 
for i in range(len(results)):
    for scores, boxes, classes, image_id, ratios in zip(*results[i]):
        #image_id = image_id
        if image_id in processed_ids:
            continue
        processed_ids.append(image_id)

        #keep = (scores > 0).nonzero()
        #scores = scores[keep].view(-1)
        #boxes = boxes[keep, :].view(-1, 4) / ratios
        #classes = classes[keep].view(-1).int()

        keep = np.transpose(np.asarray((scores > 0).nonzero()))
        scores = scores[keep].reshape(-1)
        boxes = boxes[keep, :].reshape(-1, 4) / ratios
        classes = classes[keep].reshape(-1).astype(np.int32)

        for score, box, cat in zip(scores, boxes, classes):
            x1, y1, x2, y2 = box.data.tolist()
            cat = cat.item()
            if 'annotations' in coco.dataset:
                cat = coco.getCatIds()[cat]
            this_det = {
                'image_id': image_id,
                'score': score.item(),
                'category_id': cat}

            this_det['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            detections.append(this_det)
#print(detections)
if detections:
    # Save detections
    detections = {'annotations': detections}
    detections['images'] = coco.dataset['images']
    if 'categories' in coco.dataset:
        detections['categories'] = [coco.dataset['categories']]

    # Evaluate model on dataset
    if 'annotations' in coco.dataset:
        coco_pred = coco.loadRes(detections['annotations'])
        coco_eval = COCOeval(coco, coco_pred, 'bbox')
        coco_eval.params.imgIds = processed_ids#[image_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
