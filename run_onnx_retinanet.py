#import numpy as np 
import onnxruntime
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval




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
        classes = (indices / width / height) % num_classes
        classes = classes.type(all_cls_head.type())

        # Infer kept bboxes
        x = indices % width
        y = (indices / width).type(torch.LongTensor) % height
        a = (((indices / num_classes).type(torch.LongTensor) / height).type(torch.LongTensor) / width).type(torch.LongTensor)
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
    # Inference post-processing
    anchors = {}
    decoded = []

    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        print(cls_head.shape)
        stride = image.shape[-2] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
        # Decode and filter boxes
        decoded.append(decode(cls_head, box_head, stride,
                              threshold=0.05, top_n=10, anchors=anchors[stride]))

    # Perform non-maximum suppression
    decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
    # NMS threshold = 0.5
    scores, boxes, labels = nms(*decoded, nms=0.5, ndetections=100)
    return scores, boxes, labels

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
    return img

#filename = "D:\Programs\work_dir\\0000000397133.jpg" #demo.jpg"
filename = "D:\Programs\work_dir\\val2017\\000000135604.jpg" 
model_dir = 'D:\Programs\work_dir\\test_retinanet_resnet101\\retinanet-9.onnx'
json_path = "D:\Programs\work_dir\\instances_val2017.json"
input_image = Image.open(filename).convert("RGB")
input_image = resize_im(input_image)
in_w, in_h = input_image.size
print("image size：", input_image.size)
input_image = np.array(input_image)
preprocess1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess1(input_image)
input_tensor = input_tensor.unsqueeze(0)
print("input_tesnor:", input_tensor.size())

def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]




pw = 640 - in_w
ph = 480 - in_h
input_tensor = F.pad(input_tensor, (0, pw, 0, ph))
# Test exported model with TensorProto data saved in files
inputs_flatten = flatten(input_tensor.detach().cpu().numpy())
print(inputs_flatten[0].shape)
inputs_flatten = update_flatten_list(inputs_flatten, [])
print(inputs_flatten[0].shape)

ratios = 1
ids = [135604] #[46252]

img = preprocess(input_image)
print(inputs_flatten[0].shape)
outputs = []
ort_inference(model_dir, inputs_flatten, outputs)
print(outputs[0][0].shape)
cls_heads = []
box_heads = []
for i in range(len(outputs[0])):
    if i < 5:
        cls_heads.append(torch.from_numpy(outputs[0][i]))
    else:
        box_heads.append(torch.from_numpy(outputs[0][i]))

#print("cls_heads:", cls_heads.shape)
#print("box_heads:", box_heads)
results = []
scores, boxes, classes = detection_postprocess(input_image, cls_heads, box_heads)
#print("scores:", scores)
#print("boxes:", boxes)
#print("labels:", classes)
results.append([scores, boxes, classes, ids])


detections = []
processed_ids = set()
coco = COCO(json_path) 

for scores, boxes, classes, image_id in zip(*results[0]):
    #image_id = image_id
    if image_id in processed_ids:
        continue
    processed_ids.add(image_id)

    keep = (scores > 0).nonzero()
    scores = scores[keep].view(-1)
    boxes = boxes[keep, :].view(-1, 4) / ratios
    classes = classes[keep].view(-1).int()

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
print(detections)
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
        coco_eval.params.imgIds = [image_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
