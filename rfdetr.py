import os
import json
import random
import hashlib
from pathlib import Path
from urllib.parse import urlparse, unquote
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from PIL import Image
from tqdm.auto import tqdm
from rfdetr import RFDETRBase

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_JSON_DIR = "./coco_files"
DATASET_DIR = "./rfdetr_dataset"
OUTPUT_DIR = "./rfdetr_output"
PRETRAINED_WEIGHTS = "./weights/rf-detr-base.pth"
AZURE_CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
BBOX_FORMAT = "xywh"
IMAGE_FIELD = "image_id"
CATEGORY_FIELD = "category_id"
BBOX_FIELD = "bbox"
AREA_FIELD = "area"
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
DOWNLOAD_WORKERS = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================
# PATHS
# ============================================================
INPUT_JSON_DIR = Path(INPUT_JSON_DIR)
DATASET_DIR = Path(DATASET_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)
if PRETRAINED_WEIGHTS:
    PRETRAINED_WEIGHTS = Path(PRETRAINED_WEIGHTS)
else:
    PRETRAINED_WEIGHTS = None

# ============================================================
# VALIDATION
# ============================================================
split_ratio = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
if abs(split_ratio - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1.0")
if BBOX_FORMAT not in {"xywh", "xyxy"}:
    raise ValueError("BBOX_FORMAT must be 'xywh' or 'xyxy'")
random.seed(RANDOM_SEED)

# ============================================================
# AZURE
# ============================================================
load_dotenv()
connection_string = os.getenv(AZURE_CONNECTION_STRING_ENV)
if not connection_string:
    raise EnvironmentError(f"Environment variable '{AZURE_CONNECTION_STRING_ENV}' was not found.")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# ============================================================
# HELPERS
# ============================================================
def load_annotation_file(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "annotations" in data:
            return data["annotations"]
        if "data" in data:
            return data["data"]
        return [data]
    raise ValueError(f"Unsupported JSON structure: {json_path}")

def normalize_category(category_value):
    if isinstance(category_value, list):
        if len(category_value) != 1:
            raise ValueError(f"Expected exactly one category per annotation. Received: {category_value}")
        return str(category_value[0])
    return str(category_value)

def parse_blob_url(image_url):
    parsed_url = urlparse(image_url)
    blob_path = unquote(parsed_url.path)
    blob_path = blob_path.lstrip("/")
    path_parts = blob_path.split("/", 1)
    if len(path_parts) != 2:
        raise ValueError(f"Unable to parse Azure Blob URL: {image_url}")
    container_name = path_parts[0]
    blob_name = path_parts[1]
    return container_name, blob_name

def make_local_filename(image_url):
    parsed_url = urlparse(image_url)
    decoded_path = unquote(parsed_url.path)
    original_filename = Path(decoded_path).name
    extension = Path(original_filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        extension = ".jpg"
    url_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()
    short_hash = url_hash[:12]
    return f"{short_hash}{extension}"

def convert_bbox(bbox):
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    first_value = float(bbox[0])
    second_value = float(bbox[1])
    third_value = float(bbox[2])
    fourth_value = float(bbox[3])
    if BBOX_FORMAT == "xywh":
        x = first_value
        y = second_value
        width = third_value
        height = fourth_value
        return [x, y, width, height]
    x1 = first_value
    y1 = second_value
    x2 = third_value
    y2 = fourth_value
    width = x2 - x1
    height = y2 - y1
    return [x1, y1, width, height]

def download_blob_image(image_url, destination):
    if destination.exists():
        return destination
    container_name, blob_name = parse_blob_url(image_url)
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as file:
        download_stream = blob_client.download_blob()
        download_stream.readinto(file)
    return destination

def get_image_size(image_path):
    with Image.open(image_path) as image:
        width = image.width
        height = image.height
    return width, height

# ============================================================
# READ JSON FILES
# ============================================================
print("\n" + "=" * 70)
print("READING ANNOTATION FILES")
print("=" * 70)
json_files = sorted(INPUT_JSON_DIR.glob("*.json"))
if not json_files:
    raise FileNotFoundError(f"No JSON files found in {INPUT_JSON_DIR}")
print(f"JSON files found: {len(json_files):,}")
all_records = []
for json_file in tqdm(json_files, desc="Reading JSON files", unit="file"):
    records = load_annotation_file(json_file)
    all_records.extend(records)
print(f"Total annotations loaded: {len(all_records):,}")

# ============================================================
# PROCESS ANNOTATIONS
# ============================================================
print("\n" + "=" * 70)
print("PROCESSING ANNOTATIONS")
print("=" * 70)
classes = []
class_set = set()
image_records = defaultdict(list)
invalid_annotations = 0
for record in tqdm(all_records, desc="Processing annotations", unit="annotation"):
    if IMAGE_FIELD not in record:
        invalid_annotations += 1
        continue
    if CATEGORY_FIELD not in record:
        invalid_annotations += 1
        continue
    if BBOX_FIELD not in record:
        invalid_annotations += 1
        continue
    try:
        image_url = str(record[IMAGE_FIELD])
        category_name = normalize_category(record[CATEGORY_FIELD])
        bbox = convert_bbox(record[BBOX_FIELD])
        bbox_width = bbox[2]
        bbox_height = bbox[3]
        if bbox_width <= 0:
            invalid_annotations += 1
            continue
        if bbox_height <= 0:
            invalid_annotations += 1
            continue
        if category_name not in class_set:
            class_set.add(category_name)
            classes.append(category_name)
        image_records[image_url].append({
            "category": category_name,
            "bbox": bbox
        })
    except Exception:
        invalid_annotations += 1

valid_annotation_count = sum(
    len(records)
    for records in image_records.values()
)
print(f"Valid annotations: {valid_annotation_count:,}")
print(f"Invalid annotations: {invalid_annotations:,}")

# ============================================================
# AUTO-DETECT CLASSES
# ============================================================
print("\n" + "=" * 70)
print("AUTO-DETECTED CLASSES")
print("=" * 70)
category_to_id = {}
for index, class_name in enumerate(
    tqdm(classes, desc="Creating class mapping", unit="class"),
    start=1
):
    category_to_id[class_name] = index
for class_name, class_id in category_to_id.items():
    print(f"{class_id:3d} -> {class_name}")
number_of_classes = len(category_to_id)
if number_of_classes == 0:
    raise ValueError("No classes were detected.")

# ============================================================
# DATASET SPLIT
# ============================================================
print("\n" + "=" * 70)
print("CREATING DATASET SPLIT")
print("=" * 70)
image_urls = list(image_records.keys())
random.shuffle(image_urls)
total_images = len(image_urls)
train_count = int(total_images * TRAIN_RATIO)
valid_count = int(total_images * VALID_RATIO)
train_urls = image_urls[:train_count]
valid_urls = image_urls[train_count:train_count + valid_count]
test_urls = image_urls[train_count + valid_count:]
splits = {
    "train": train_urls,
    "valid": valid_urls,
    "test": test_urls
}
for split_name, urls in tqdm(
    splits.items(),
    desc="Preparing dataset splits",
    unit="split"
):
    print(f"{split_name:>6}: {len(urls):,} images")

# ============================================================
# CREATE DIRECTORIES
# ============================================================
print("\n" + "=" * 70)
print("CREATING DATASET DIRECTORIES")
print("=" * 70)
for split_name in tqdm(
    splits.keys(),
    desc="Creating directories",
    unit="split"
):
    split_directory = DATASET_DIR / split_name
    image_directory = split_directory / "images"
    image_directory.mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# DOWNLOAD IMAGES
# ============================================================
print("\n" + "=" * 70)
print("DOWNLOADING IMAGES FROM AZURE")
print("=" * 70)

def download_split(split_name, urls):
    split_directory = DATASET_DIR / split_name / "images"
    results = {}
    futures = {}
    already_existing = 0
    failed_downloads = 0
    executor = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    try:
        for image_url in tqdm(
            urls,
            desc=f"Preparing {split_name} downloads",
            unit="image"
        ):
            filename = make_local_filename(image_url)
            destination = split_directory / filename
            if destination.exists():
                results[image_url] = filename
                already_existing += 1
                continue
            future = executor.submit(
                download_blob_image,
                image_url,
                destination
            )
            futures[future] = (
                image_url,
                filename
            )
        with tqdm(
            total=len(futures),
            desc=f"Downloading {split_name}",
            unit="image"
        ) as progress:
            for future in as_completed(futures):
                image_url, filename = futures[future]
                try:
                    future.result()
                    results[image_url] = filename
                except Exception as error:
                    failed_downloads += 1
                    print(f"\nDownload failed: {image_url}")
                    print(f"Error: {error}")
                progress.update(1)
    finally:
        executor.shutdown(wait=True)
    print(f"{split_name}: {len(results):,} available")
    print(f"{split_name}: {already_existing:,} already existed")
    print(f"{split_name}: {failed_downloads:,} failed")
    return results

local_image_map = {}
for split_name, urls in tqdm(
    splits.items(),
    desc="Processing dataset splits",
    unit="split"
):
    local_image_map[split_name] = download_split(
        split_name,
        urls
    )

# ============================================================
# CREATE COCO ANNOTATIONS
# ============================================================
print("\n" + "=" * 70)
print("CREATING COCO ANNOTATIONS")
print("=" * 70)

def create_coco_json(split_name, urls):
    split_directory = DATASET_DIR / split_name
    image_directory = split_directory / "images"
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    image_id = 1
    processed_images = 0
    processed_annotations = 0
    for image_url in tqdm(
        urls,
        desc=f"Building {split_name} COCO",
        unit="image"
    ):
        if image_url not in local_image_map[split_name]:
            continue
        filename = local_image_map[split_name][image_url]
        image_path = image_directory / filename
        if not image_path.exists():
            continue
        try:
            image_width, image_height = get_image_size(image_path)
        except Exception as error:
            print(f"\nUnable to read image: {image_path}")
            print(f"Error: {error}")
            continue
        coco_images.append({
            "id": image_id,
            "file_name": f"images/{filename}",
            "width": image_width,
            "height": image_height
        })
        image_annotations = image_records[image_url]
        for item in tqdm(
            image_annotations,
            desc=f"Annotations for {filename}",
            unit="bbox",
            leave=False
        ):
            category_name = item["category"]
            bbox = item["bbox"]
            x = bbox[0]
            y = bbox[1]
            width = bbox[2]
            height = bbox[3]
            if x < 0:
                x = 0.0
            if y < 0:
                y = 0.0
            maximum_width = image_width - x
            maximum_height = image_height - y
            if maximum_width < 0:
                maximum_width = 0.0
            if maximum_height < 0:
                maximum_height = 0.0
            width = min(
                width,
                maximum_width
            )
            height = min(
                height,
                maximum_height
            )
            if width <= 0:
                continue
            if height <= 0:
                continue
            area = width * height
            category_id = category_to_id[category_name]
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [
                    round(x, 4),
                    round(y, 4),
                    round(width, 4),
                    round(height, 4)
                ],
                "area": round(area, 4),
                "iscrowd": 0
            }
            coco_annotations.append(
                coco_annotation
            )
            annotation_id += 1
            processed_annotations += 1
        image_id += 1
        processed_images += 1
    coco_categories = []
    for category_name, category_id in tqdm(
        category_to_id.items(),
        desc=f"Creating {split_name} categories",
        unit="class"
    ):
        coco_categories.append({
            "id": category_id,
            "name": category_name,
            "supercategory": "location_tag"
        })
    coco_data = {
        "info": {
            "description": "RF-DETR Location Tag Dataset",
            "version": "1.0"
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }
    annotation_path = split_directory / "_annotations.coco.json"
    with open(
        annotation_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            coco_data,
            file,
            indent=2
        )
    print(f"{split_name}: {processed_images:,} images")
    print(f"{split_name}: {processed_annotations:,} annotations")
    print(f"COCO file: {annotation_path}")
    return annotation_path

annotation_paths = {}
for split_name, urls in tqdm(
    splits.items(),
    desc="Creating COCO datasets",
    unit="split"
):
    annotation_paths[split_name] = create_coco_json(
        split_name,
        urls
    )

# ============================================================
# SAVE CLASS MAPPING
# ============================================================
print("\n" + "=" * 70)
print("SAVING CLASS MAPPING")
print("=" * 70)
class_mapping_path = DATASET_DIR / "classes.json"
with open(
    class_mapping_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        category_to_id,
        file,
        indent=2
    )
print(f"Class mapping saved: {class_mapping_path}")

# ============================================================
# DATASET SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"Number of classes : {number_of_classes:,}")
print(f"Total images      : {total_images:,}")
print(f"Train images      : {len(train_urls):,}")
print(f"Valid images      : {len(valid_urls):,}")
print(f"Test images       : {len(test_urls):,}")
print(f"Annotations       : {valid_annotation_count:,}")
print(f"BBOX format       : {BBOX_FORMAT}")
print(f"Dataset directory : {DATASET_DIR}")

# ============================================================
# PRETRAINED WEIGHTS
# ============================================================
print("\n" + "=" * 70)
print("CHECKING PRETRAINED WEIGHTS")
print("=" * 70)
if PRETRAINED_WEIGHTS is not None:
    print(f"Pretrained weights: {PRETRAINED_WEIGHTS}")
    if not PRETRAINED_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Pretrained weights were not found: {PRETRAINED_WEIGHTS}"
        )
    pretrained_size = PRETRAINED_WEIGHTS.stat().st_size
    pretrained_size_mb = pretrained_size / (1024 * 1024)
    print(f"Weight file size: {pretrained_size_mb:.2f} MB")
else:
    print("No pretrained weights configured.")

# ============================================================
# CREATE MODEL
# ============================================================
print("\n" + "=" * 70)
print("CREATING RF-DETR MODEL")
print("=" * 70)
if PRETRAINED_WEIGHTS is not None:
    model = RFDETRBase(
        pretrain_weights=str(
            PRETRAINED_WEIGHTS
        )
    )
    print("RF-DETR initialized with pretrained weights.")
else:
    model = RFDETRBase()
    print("RF-DETR initialized without pretrained weights.")

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# TRAIN
# ============================================================
print("\n" + "=" * 70)
print("STARTING RF-DETR TRAINING")
print("=" * 70)
print(f"Dataset directory : {DATASET_DIR}")
print(f"Output directory  : {OUTPUT_DIR}")
print(f"Epochs            : {EPOCHS}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Grad accumulation : {GRAD_ACCUM_STEPS}")
print("RF-DETR training started...")
model.train(
    dataset_dir=str(
        DATASET_DIR
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    output_dir=str(
        OUTPUT_DIR
    )
)
print("RF-DETR training completed.")

# ============================================================
# FIND BEST CHECKPOINT
# ============================================================
print("\n" + "=" * 70)
print("SEARCHING FOR BEST CHECKPOINT")
print("=" * 70)
possible_checkpoints = [
    OUTPUT_DIR / "checkpoint_best.pth",
    OUTPUT_DIR / "checkpoint_best.pt",
    OUTPUT_DIR / "best.pth",
    OUTPUT_DIR / "best.pt"
]
best_checkpoint = None
for checkpoint in tqdm(
    possible_checkpoints,
    desc="Checking checkpoints",
    unit="file"
):
    if checkpoint.exists():
        best_checkpoint = checkpoint
        break
if best_checkpoint is not None:
    print(f"Best checkpoint found: {best_checkpoint}")
else:
    print("Best checkpoint was not found automatically.")

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("STARTING EVALUATION")
print("=" * 70)
try:
    evaluation_model = model
    if best_checkpoint is not None:
        print("Loading best checkpoint for evaluation...")
        evaluation_model = RFDETRBase(
            pretrain_weights=str(
                best_checkpoint
            )
        )
    print("Running RF-DETR evaluation...")
    evaluation_result = evaluation_model.evaluate(
        dataset_dir=str(
            DATASET_DIR
        )
    )
    print("\n" + "=" * 70)
    print("EVALUATION RESULT")
    print("=" * 70)
    print(evaluation_result)
except Exception as error:
    print("\n" + "=" * 70)
    print("EVALUATION FAILED")
    print("=" * 70)
    print(str(error))

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("RF-DETR PIPELINE COMPLETED")
print("=" * 70)
print(f"Classes            : {classes}")
print(f"Number of classes  : {number_of_classes}")
print(f"Total images       : {total_images}")
print(f"Train images       : {len(train_urls)}")
print(f"Valid images       : {len(valid_urls)}")
print(f"Test images        : {len(test_urls)}")
print(f"Annotations        : {valid_annotation_count}")
print(f"BBOX format        : {BBOX_FORMAT}")
print(f"Pretrained weights : {PRETRAINED_WEIGHTS}")
print(f"Dataset directory  : {DATASET_DIR}")
print(f"Output directory   : {OUTPUT_DIR}")
print("=" * 70)
