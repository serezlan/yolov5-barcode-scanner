#!/usr/bin/env python3

import argparse
import numpy as np
import cv2


def parse_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model", help="YOLO v5 model", default="best.onnx")
    parser.add_argument(
        "-l",
        "--classlist",
        help="config file describing class list",
        default="class-list.txt",
    )
    parser.add_argument("image", help="input image file")
    # parser.add_argument( "-c", "--confidence", help="confidence threshold", default=0.4)

    args = parser.parse_args()
    config = vars(args)
    print(config)
    return config


def format_yolov5(frame):
    """ "Reshape input to 640x640 rectangular image."""

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def load_model(model_file):
    """Load model"""

    return cv2.dnn.readNet(model_file)


def get_prediction(net, image):
    """ "Run prediction on image"""

    input_image = format_yolov5(image)  # making the image square
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    return net.forward()


def unwrap_prediction(predictions, input_image):
    """Select predictions above expected threshold and generate bounding along with confidence levels"""

    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > 0.25:
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    return (class_ids, boxes, confidences)


def get_class_list(config_file):
    """get list of class for classification"""

    with open(config_file, "r") as f:
        return [cname.strip() for cname in f.readlines()]


def post_process(class_list, class_ids, image, boxes, confidences):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):
        box = result_boxes[i]
        class_id = result_class_ids[i]

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(
            image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1
        )
        cv2.putText(
            image,
            class_list[class_id],
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )


def main():

    args = parse_arguments()
    model = load_model(args["model"])
    class_list = get_class_list(args["classlist"])
    video = cv2.VideoCapture(0)

    while True:
        ret, image = video.read()
        predictions = get_prediction(model, image)
        class_ids, boxes, confidences = unwrap_prediction(predictions, image)
        post_process(class_list, class_ids, image, boxes, confidences)

        cv2.imshow("output", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
