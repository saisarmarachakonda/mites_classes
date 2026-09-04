# ============================================================
# RF-DETR ENVIRONMENT + MODEL + DATASET EXPECTATION INSPECTION
# ============================================================

import os
import inspect
import json
from pathlib import Path

print("=" * 90)
print("RF-DETR EXPECTATION INSPECTION")
print("=" * 90)

# ------------------------------------------------------------
# 1. RF-DETR package information
# ------------------------------------------------------------

import rfdetr

print("\n[1] RF-DETR PACKAGE")
print("-" * 90)

print("RF-DETR module:")
print(rfdetr)

print("RF-DETR location:")
print(inspect.getfile(rfdetr))


# ------------------------------------------------------------
# 2. RF-DETR version
# ------------------------------------------------------------

print("\n[2] RF-DETR VERSION")
print("-" * 90)

try:
    from importlib.metadata import version

    print("Version:", version("rfdetr"))

except Exception as e:
    print("Could not determine version:", e)


# ------------------------------------------------------------
# 3. RFDETRBase constructor
# ------------------------------------------------------------

from rfdetr import RFDETRBase

print("\n[3] RFDETRBase CONSTRUCTOR")
print("-" * 90)

print(inspect.signature(RFDETRBase))

print("\nConstructor source:")
print(inspect.getsource(RFDETRBase.__init__))


# ------------------------------------------------------------
# 4. Load pretrained model
# ------------------------------------------------------------

PRETRAINED_WEIGHTS = "./weights/rf-detr-base.pth"

print("\n[4] PRETRAINED MODEL")
print("-" * 90)

print("Weights:")
print(Path(PRETRAINED_WEIGHTS).resolve())

print("Exists:")
print(Path(PRETRAINED_WEIGHTS).exists())

if not Path(PRETRAINED_WEIGHTS).exists():
    raise FileNotFoundError(
        f"Pretrained weights not found: {PRETRAINED_WEIGHTS}"
    )

model = RFDETRBase(
    pretrain_weights=PRETRAINED_WEIGHTS
)

print("\n✓ Model loaded successfully.")
print("Model type:", type(model))


# ------------------------------------------------------------
# 5. Model attributes
# ------------------------------------------------------------

print("\n[5] LOADED MODEL ATTRIBUTES")
print("-" * 90)

model_attributes = [
    name
    for name in dir(model)
    if not name.startswith("_")
]

for name in model_attributes:
    print(name)


# ------------------------------------------------------------
# 6. Find configuration objects
# ------------------------------------------------------------

print("\n[6] CONFIGURATION OBJECTS")
print("-" * 90)

configuration_objects = {}

for name in model_attributes:

    try:
        value = getattr(model, name)

        if (
            "config" in name.lower()
            or "config" in type(value).__name__.lower()
        ):
            configuration_objects[name] = value

            print("\nATTRIBUTE:", name)
            print("TYPE:", type(value))

            if hasattr(value, "__dict__"):
                for key, val in vars(value).items():
                    print(f"  {key:35} : {val}")
            else:
                print("VALUE:", value)

    except Exception as e:
        print(f"{name}: unable to inspect ({e})")


# ------------------------------------------------------------
# 7. Underlying model
# ------------------------------------------------------------

print("\n[7] UNDERLYING MODEL")
print("-" * 90)

if hasattr(model, "model"):

    underlying_model = model.model

    print("Type:")
    print(type(underlying_model))

    print("\nModel attributes:")

    for name in dir(underlying_model):

        if not name.startswith("_"):
            print(name)

else:

    print("model.model not available.")


# ------------------------------------------------------------
# 8. Model configuration values
# ------------------------------------------------------------

print("\n[8] IMPORTANT MODEL SETTINGS")
print("-" * 90)

IMPORTANT_SETTINGS = [
    "num_classes",
    "num_queries",
    "resolution",
    "hidden_dim",
    "encoder",
    "dec_layers",
    "two_stage",
    "group_detr",
    "projector_scale",
    "num_channels",
    "amp",
    "device",
    "gradient_checkpointing",
    "compile",
    "pretrain_weights",
]

for config_name, config_object in configuration_objects.items():

    print("\nCONFIG:", config_name)

    for setting in IMPORTANT_SETTINGS:

        if hasattr(config_object, setting):

            value = getattr(
                config_object,
                setting
            )

            print(
                f"{setting:30} : {value}"
            )


# ------------------------------------------------------------
# 9. RF-DETR COCO dataset implementation
# ------------------------------------------------------------

print("\n[9] RF-DETR COCO DATASET IMPLEMENTATION")
print("-" * 90)

from rfdetr.datasets.coco import CocoDetection

print("CocoDetection class:")
print(CocoDetection)

print("\nCocoDetection source:")
print(inspect.getsource(CocoDetection))


# ------------------------------------------------------------
# 10. Dataset directory
# ------------------------------------------------------------

print("\n[10] DATASET DIRECTORY")
print("-" * 90)

DATASET_DIR = Path("./rfdetr_dataset")

print("Current working directory:")
print(Path.cwd())

print("\nDataset directory:")
print(DATASET_DIR.resolve())

print("\nDataset exists:")
print(DATASET_DIR.exists())


# ------------------------------------------------------------
# 11. Inspect train/valid/test structure
# ------------------------------------------------------------

print("\n[11] DATASET STRUCTURE")
print("-" * 90)

for split in ["train", "valid", "test"]:

    split_dir = DATASET_DIR / split

    print(f"\n{split.upper()}")
    print("-" * 50)

    print("Directory:")
    print(split_dir.resolve())

    print("Exists:")
    print(split_dir.exists())

    if split_dir.exists():

        for item in split_dir.iterdir():
            print(" ", item)


# ------------------------------------------------------------
# 12. Inspect COCO JSON
# ------------------------------------------------------------

print("\n[12] COCO ANNOTATION STRUCTURE")
print("-" * 90)

for split in ["train", "valid", "test"]:

    coco_file = (
        DATASET_DIR
        / split
        / "_annotations.coco.json"
    )

    print(f"\n{split.upper()}")
    print("COCO file:", coco_file.resolve())
    print("Exists:", coco_file.exists())

    if not coco_file.exists():
        continue

    with coco_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        coco = json.load(f)

    print("Images:", len(coco.get("images", [])))
    print("Annotations:", len(coco.get("annotations", [])))
    print("Categories:", len(coco.get("categories", [])))

    print("\nFirst image entry:")

    if coco.get("images"):
        print(
            json.dumps(
                coco["images"][0],
                indent=2
            )
        )

    print("\nCategories:")

    for category in coco.get("categories", []):
        print(category)


# ------------------------------------------------------------
# 13. Calculate actual paths from COCO
# ------------------------------------------------------------

print("\n[13] COCO PATH EXPECTATION")
print("-" * 90)

for split in ["train", "valid", "test"]:

    split_dir = DATASET_DIR / split

    coco_file = (
        split_dir
        / "_annotations.coco.json"
    )

    if not coco_file.exists():
        continue

    with coco_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        coco = json.load(f)

    print(f"\n{split.upper()}")

    for image in coco.get("images", [])[:5]:

        file_name = image["file_name"]

        expected_path = (
            split_dir
            / file_name
        )

        images_path = (
            split_dir
            / "images"
            / Path(file_name).name
        )

        print("\nCOCO file_name:")
        print(file_name)

        print("\nroot + file_name:")
        print(expected_path.resolve())

        print(
            "Exists:",
            expected_path.exists()
        )

        print("\nimages/ + filename:")
        print(images_path.resolve())

        print(
            "Exists:",
            images_path.exists()
        )


# ------------------------------------------------------------
# 14. Dataset class count
# ------------------------------------------------------------

print("\n[14] DATASET CLASS COUNT")
print("-" * 90)

train_coco_file = (
    DATASET_DIR
    / "train"
    / "_annotations.coco.json"
)

if train_coco_file.exists():

    with train_coco_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        train_coco = json.load(f)

    categories = train_coco.get(
        "categories",
        []
    )

    print("Number of classes:", len(categories))

    for category in categories:
        print(
            category["id"],
            ":",
            category["name"]
        )


# ------------------------------------------------------------
# 15. Training API
# ------------------------------------------------------------

print("\n[15] RF-DETR TRAINING API")
print("-" * 90)

print(
    inspect.signature(model.train)
)

print("\nTraining source:")

try:
    print(inspect.getsource(model.train))
except Exception as e:
    print("Could not retrieve training source:", e)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("INSPECTION COMPLETE")
print("=" * 90)

print(
    """
IMPORTANT:
No training was started.
No dataset files were modified.
No COCO files were modified.

Use this output to determine the exact expectations
of the installed RF-DETR version and pretrained model.
"""
)
