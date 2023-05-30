from models.Metrics import Metrics
from ultralytics import YOLO
from clip import clip
from models.CustomDataset import CustomDataset
import utils
import torch


if __name__ == "__main__":
    yolo_model = YOLO("yolov8n.pt")

    clip_model, preprocess = clip.load("RN50")
    clip_model = clip_model.eval()

    train_dataset = CustomDataset(split="train", shuffle=True)
    val_dataset = CustomDataset(split="val", shuffle=True)
    test_dataset = CustomDataset(split="test")

    metrics = Metrics(
        iou_threshold=0.5, prob_threshold=0.5, dataset_dim=len(val_dataset)
    )

    for image, random_sentence, ground_bbox, category_id in val_dataset:
        results = yolo_model(image)
        boxes = results[0].boxes

        if boxes is None:
            metrics.update_metrics(no_predictions=True)
        else:
            cropped_images = utils.crop_image_by_boxes(image, boxes)
            preprocessed_images = utils.preprocess_images(cropped_images, preprocess)
            images_z, texts_z = utils.encode_data_with_clip(
                clip_model, preprocessed_images, random_sentence
            )

            # print_cosine_similarity_matrix(images_z, texts_z)

            probs = utils.get_probs(images_z, texts_z)

            # threshold = utils.get_threshold(probs)

            # print(random_sentence)
            # print(threshold)
            # print(probs)

            obj_index = torch.argmax(probs)
            print(f"obj_index: {obj_index}")
            # utils.plot_bounding_boxes(image, boxes, obj_indeces, ground_bbox)

            ground_bbox_xyxy = utils.xywh_to_xyxy(ground_bbox)
            for index, xyxy in enumerate(boxes.xyxy):
                if index == obj_index:
                    metrics.update_metrics(
                        xyxy, ground_bbox_xyxy, prob=probs[obj_index], verbose=True
                    )

                    # print(f"x0{xywh[0]} y0{xywh[1]} x1{xywh[2]} y1{xywh[3]}")
                    # print(f"x0{ground_bbox_xyxy[0]} y0{ground_bbox_xyxy[1]} x1{ground_bbox_xyxy[2]} y1{ground_bbox_xyxy[3]}")
                    # _, ax = plt.subplots()
                    # plt.imshow(image)
                    # plt.scatter(xywh[0], xywh[1], c="g")
                    # plt.scatter(xywh[2], xywh[3], c="g")
                    # plt.scatter(ground_bbox_xyxy[0], ground_bbox_xyxy[1], c="r")
                    # plt.scatter(ground_bbox_xyxy[2], ground_bbox_xyxy[3], c="r")
                    # plt.show()

    metrics.compute_final_iou()
    metrics.compute_precision()
    metrics.compute_recall()
    metrics.compute_f1_score()
    metrics.print_metrics()
