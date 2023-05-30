class Metrics:
    def __init__(self, iou_threshold, prob_threshold, dataset_dim):
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.dataset_dim = dataset_dim
        self.iou = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.annotations_analized_so_far = 0

    def update_metrics(
        self,
        predicted_boxes=None,
        ground_boxes=None,
        prob=0.0,
        verbose=False,
        no_predictions=False,
    ):
        if no_predictions:
            self.FN += 1
            return

        iou = self._compute_iou(predicted_boxes, ground_boxes)

        if verbose:
            print(f"Annotations analized: {self.annotations_analized_so_far}")
            print("+----------------------+")
            print(f"IOU: {iou}")
            
        if iou > self.iou_threshold and prob > self.prob_threshold:
            self.TP += 1
        elif iou <= self.iou_threshold and prob > self.prob_threshold:
            self.FP += 1
        elif prob <= self.prob_threshold:
            self.FN += 1

        self.iou += iou
        self.annotations_analized_so_far += 1

    def print_metrics(self):
        print(f"Annotations analized: {self.annotations_analized_so_far}")
        print("+----------------------+")
        print(f"IOU: {self.iou}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1 Score: {self.f1_score}")
        print(f"TP: {self.TP}")
        print(f"FP: {self.FP}")
        print(f"FN: {self.FN}")

    def _compute_iou(self, predicted_box, ground_box):
        x1 = max(predicted_box[0], ground_box[0])
        y1 = max(predicted_box[1], ground_box[1])
        x2 = min(predicted_box[2], ground_box[2])
        y2 = min(predicted_box[3], ground_box[3])
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        box1_area = (predicted_box[2] - predicted_box[0] + 1) * (
            predicted_box[3] - predicted_box[1] + 1
        )
        box2_area = (ground_box[2] - ground_box[0] + 1) * (
            ground_box[3] - ground_box[1] + 1
        )

        # Calculate the IoU

        if intersection_area > 0:
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            return iou
        else:
            return 0.0

    def compute_final_iou(self):
        self.iou = self.iou / self.dataset_dim

    def compute_precision(self):
        self.precision = self.TP / (self.TP + self.FP)

    def compute_recall(self):
        self.recall = self.TP / (self.TP + self.FN)

    def compute_f1_score(self):
        self.f1_score = (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
        )

