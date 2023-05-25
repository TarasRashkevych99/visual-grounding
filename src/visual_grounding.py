from ultralytics import YOLO
from clip import clip
import utils
import matplotlib.pyplot as plt
from utils import plot_ground_bbox
import json


if __name__ == "__main__":
    yolo_model = YOLO("yolov8n.pt")

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    train, val, test = utils.get_partitions()
    bbox = json.load(open("refcocog/annotations/instances_bbox.json"))
    count = 0
    for image, random_sentence, image_name in train:
        count += 1
        print(count)
        # print(image_name)
        # plot_ground_bbox(image_name, bbox)
        # results = yolo_model(image)
        # boxes = results[0].boxes

        # cropped_images = utils.crop_image_by_boxes(image, boxes)
        # preprocessed_images = utils.preprocess_images(cropped_images, preprocess)
        # images_z, texts_z = utils.encode_data_with_clip(
        #     clip_model, preprocessed_images, random_sentence
        # )

        # # print_cosine_similarity_matrix(images_z, texts_z)

        # probs = utils.get_probs(images_z, texts_z)

        # threshold = utils.get_threshold(probs)

        # print(random_sentence)
        # print(threshold)
        # print(probs)

        # obj_indeces = [
        #     index for (index, prob) in enumerate(probs[:]) if prob > threshold
        # ]

        # utils.plot_bounding_boxes(image, boxes, obj_indeces, image_name)