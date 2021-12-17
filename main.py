import cv2
import os
import json


def label_image(image_path):
    """
    :param image_path:
    :return:
    """
    img = cv2.imread(image_path)
    image_name = os.path.split(image_path)[-1].split(".")[0]
    with open('bin/coco.names', 'r') as f:
        classes = f.read().splitlines()

    net = cv2.dnn.readNetFromDarknet('bin/yolov4.cfg', 'bin/yolov4.weights')

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)

        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    color=(0, 255, 0), thickness=10)

    if not os.path.exists("Results"):
        os.makedirs("Results")
    cv2.imwrite(f"Results/{image_name}output.jpg", img)


def validate_images():
    with open('bin/coco.names', 'r') as f:
        classes = f.read().splitlines()

    net = cv2.dnn.readNetFromDarknet('bin/yolov4.cfg', 'bin/yolov4.weights')

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    quickstart_path = "fiftyone/quickstart/"
    with open(os.path.join(quickstart_path, "coco.json")) as f:
        coco = json.load(f)

    annotations = {image['id']:[] for image in coco['images']}
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)
    for image in coco['images']:
        image_path = os.path.join(quickstart_path,"data",image['file_name'])
        img = cv2.imread(image_path)
        image_name = os.path.split(image_path)[-1].split(".")[0]
        classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

        for (classId, score, box) in zip(classIds, scores, boxes):
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                          color=(0, 255, 0), thickness=1)

            text = '%s: %.2f' % (classes[classId], score)
            cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=1)
        for annotation in annotations[image['id']]:
            box_x = int(annotation['bbox'][0])
            box_y = int(annotation['bbox'][1])
            width = int(annotation['bbox'][2])
            height = int(annotation['bbox'][3])
            cv2.rectangle(img, (box_x, box_y),
                          (box_x + width,
                           box_y + height),
                          color=(255, 0, 0), thickness=1)
            category_id = annotation['category_id']
            text = coco['categories'][category_id]['name']
            cv2.putText(img, text, (box_x, box_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 0, 0), thickness=1)
            ...
        if not os.path.exists("Results"):
            os.makedirs("Results")
        cv2.imwrite(f"Results/{image_name}output.jpg", img)

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    foo = False
    if foo:
        mypath = "Images"
        onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        for f in onlyfiles:
            label_image(os.path.join(mypath, f))
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    else:
        # with open(os.path.join(quickstart_path, "samples.json")) as f:
        #     samples = json.load(f)['samples']
        # with open("fiftyone/quickstart/metadata.json") as f:
        #     metadata = json.load(f)
        validate_images()