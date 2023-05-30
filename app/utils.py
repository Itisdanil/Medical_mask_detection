from PIL import Image
import io
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Test_images(object):
    def __init__(self, transforms, img_path):
        self.transforms = transforms
        self.img_path = img_path

    def __getitem__(self, idx):
        img = Image.open(self.img_path).convert("RGB")

        if self.transforms is not None:
            self.img = self.transforms(img)
        return self.img

    def __len__(self):
        return len(self.img)


def model_detection(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    plt.axis('off')
    img = img_tensor.cpu().data
    ax.imshow(img.permute(1, 2, 0))

    dict_class = {1: 'no mask', 2: 'mask', 3: 'inc mask'}
    dict_color = {1: 'r', 2: 'g', 3: 'b'}
    counter = 0
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box
        predict = 'None'
        for i in range(1, 4):
            if int(annotation["labels"][counter]) == i:
                try:
                    predict = dict_class[i] + ':' + str(round(float(annotation["scores"][counter]), 2))
                    rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                             linewidth=1, edgecolor=dict_color[i], facecolor='none')
                except:
                    predict = dict_class[i]
                    rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                             linewidth=1, edgecolor=dict_color[i], facecolor='none')

        if predict != 'None':
            plt.text(xmax, ymax, f'{predict}', size=((xmax - xmin) / img.shape[2]) * 80, rotation=0.,
                     ha="right",
                     va="top",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8)))
        ax.add_patch(rect)
        counter += 1

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return buf


def area_boxes(boxA, boxB):
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))

    interArea = abs(max(0, xB - xA) * max(0, yB - yA))

    boxAArea = abs((float(boxA[2]) - float(boxA[0])) * (float(boxA[3]) - float(boxA[1])))
    boxBArea = abs((float(boxB[2]) - float(boxB[0])) * (float(boxB[3]) - float(boxB[1])))

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def clean_boxes(pred_boxes, area_boxes):
    clean_pred_boxes = {'boxes': [], 'labels': [], 'scores': []}
    final_boxes = {'boxes': [], 'labels': [], 'scores': []}
    for i in range(len(pred_boxes['boxes'])):
        if pred_boxes['scores'][i] > 0.2:
            clean_pred_boxes['boxes'].append(list(map(lambda box: int(box), pred_boxes['boxes'][i])))
            clean_pred_boxes['labels'].append(pred_boxes['labels'][i].item())
            clean_pred_boxes['scores'].append(pred_boxes['scores'][i].item())
    for i in range(len(clean_pred_boxes['boxes'])):
        max_probability = clean_pred_boxes['scores'][i]
        ind = i
        for j in range(len(clean_pred_boxes['boxes'])):
            if area_boxes(clean_pred_boxes['boxes'][i], clean_pred_boxes['boxes'][j]) > 0.3 and \
                    clean_pred_boxes['scores'][j] > max_probability:
                max_probability = clean_pred_boxes['scores'][j]
                ind = j
        if clean_pred_boxes['boxes'][ind] not in final_boxes['boxes']:
            final_boxes['boxes'].append(clean_pred_boxes['boxes'][ind])
            final_boxes['labels'].append(clean_pred_boxes['labels'][ind])
            final_boxes['scores'].append(clean_pred_boxes['scores'][ind])
    return final_boxes
