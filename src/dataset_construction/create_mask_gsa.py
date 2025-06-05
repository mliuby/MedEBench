import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
args = parser.parse_args()

start_index = args.start
end_index = args.end

import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from transformers import SamModel, SamProcessor, pipeline

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def load_image(image_str: str) -> Image.Image:
    return Image.open(image_str).convert("RGB")

def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    boxes = [result.box.xyxy for result in results]
    return [boxes]

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour.reshape(-1, 2).tolist()

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    if polygon:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float().permute(0, 2, 3, 1).mean(dim=-1)
    masks = (masks > 0).int().numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for i in range(len(masks)):
            shape = masks[i].shape
            polygon = mask_to_polygon(masks[i])
            masks[i] = polygon_to_mask(polygon, shape)

    return masks

def detect(image: Image.Image, labels: List[str], threshold: float = 0.3, detector_id: Optional[str] = None) -> List[DetectionResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    return [DetectionResult.from_dict(result) for result in results]

def segment(image: Image.Image, detection_results: List[DetectionResult], polygon_refinement: bool = False, segmenter_id: Optional[str] = None) -> List[DetectionResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    segmentator = SamModel.from_pretrained(segmenter_id).to(device)
    processor = SamProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)
    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections

# === Config and Execution ===
detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"
threshold = 0.05

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
metadata_path = os.path.join(BASE_DIR, "editing/editing_metadata.json")
output_dir = os.path.join(BASE_DIR, "editing/gsa_masks_previous")
os.makedirs(output_dir, exist_ok=True)

with open(metadata_path, "r") as f:
    data = json.load(f)["samples"]

for sample in tqdm(data[start_index:end_index], desc="Processing Masks", unit="image"):
    
    idx = sample['id']
    print(idx)
    image_path = os.path.join(BASE_DIR, os.path.normpath(sample['previous_image']))
    labels = sample['Task'] if isinstance(sample['Task'], list) else [sample['Task']]

    image_array, detections = grounded_segmentation(
        image=image_path,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )

    for i, detection in enumerate(detections):
        if detection.mask is not None:
            path = os.path.join(output_dir, f"{idx}_{i}.png")
            cv2.imwrite(path, detection.mask)


# conda activate gsa_env
# CUDA_VISIBLE_DEVICES=1 python src/create_mask_gsa.py 