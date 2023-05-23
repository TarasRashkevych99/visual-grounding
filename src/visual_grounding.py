from ultralytics import YOLO
from clip import clip
import utils


if __name__ == "__main__":
    yolo_model = YOLO("yolov8n.pt")

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    train, val, test = utils.get_partitions()

    for image, random_sentence in val:
        results = yolo_model(image)
        boxes = results[0].boxes

        cropped_images = utils.crop_image_by_boxes(image, boxes)
        preprocessed_images = utils.preprocess_images(cropped_images, preprocess)
        images_z, texts_z = utils.encode_data_with_clip(
            clip_model, preprocessed_images, random_sentence
        )

        # print_cosine_similarity_matrix(images_z, texts_z)

        probs = utils.get_probs(images_z, texts_z)

        threshold = utils.get_threshold(probs)

        print(random_sentence)
        print(threshold)
        print(probs)

        obj_indeces = [
            index for (index, prob) in enumerate(probs[:]) if prob > threshold
        ]

        utils.plot_bounding_boxes(image, boxes, obj_indeces)
