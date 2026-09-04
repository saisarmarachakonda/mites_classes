# ============================================================
# CHECK RF-DETR CUSTOM DATASET / CLASS CONFIGURATION
# DO NOT TRAIN
# ============================================================

import inspect
import rfdetr

print("=" * 90)
print("RF-DETR CUSTOM DATASET CONFIGURATION CHECK")
print("=" * 90)

# ------------------------------------------------------------
# 1. RF-DETR version
# ------------------------------------------------------------

try:
    from importlib.metadata import version
    print("\nRF-DETR version:")
    print(version("rfdetr"))
except Exception as e:
    print("Could not determine version:", e)


# ------------------------------------------------------------
# 2. Check available configuration functions
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("CONFIGURATION FUNCTIONS")
print("=" * 90)

from rfdetr import config

for name in dir(config):
    if not name.startswith("_"):
        obj = getattr(config, name)

        if callable(obj):
            print("\nFUNCTION/CLASS:", name)

            try:
                print("SIGNATURE:")
                print(inspect.signature(obj))
            except Exception:
                pass


# ------------------------------------------------------------
# 3. Inspect get_train_config
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("get_train_config")
print("=" * 90)

if hasattr(config, "get_train_config"):

    print(
        inspect.signature(
            config.get_train_config
        )
    )

    print("\nSOURCE:")
    print(
        inspect.getsource(
            config.get_train_config
        )
    )


# ------------------------------------------------------------
# 4. Inspect RF-DETR training module
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("RF-DETR TRAINING MODULE")
print("=" * 90)

try:

    import rfdetr.training as training

    print("Training module:")
    print(training)

    print("\nAvailable objects:")

    for name in dir(training):
        if not name.startswith("_"):
            print(name)

except Exception as e:

    print(
        "Could not import rfdetr.training:"
    )
    print(e)


# ------------------------------------------------------------
# 5. Inspect build_roboflow_from_coco
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("SEARCHING FOR build_roboflow_from_coco")
print("=" * 90)

found = False

modules_to_check = [
    rfdetr,
    config,
]

try:
    import rfdetr.datasets.coco as coco_module
    modules_to_check.append(coco_module)
except Exception:
    pass

try:
    import rfdetr.datasets as datasets_module
    modules_to_check.append(datasets_module)
except Exception:
    pass

for module in modules_to_check:

    for name in dir(module):

        if "roboflow" in name.lower():

            obj = getattr(module, name)

            print(
                f"\nFOUND: {module.__name__}.{name}"
            )

            try:
                print(
                    "Signature:",
                    inspect.signature(obj)
                )
            except Exception:
                pass

            try:
                print(
                    "\nSource:"
                )
                print(
                    inspect.getsource(obj)
                )
            except Exception:
                pass

            found = True


if not found:
    print(
        "build_roboflow_from_coco was not found "
        "in the checked modules."
    )


# ------------------------------------------------------------
# 6. Inspect model train signature
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("RFDETRBase.train SIGNATURE")
print("=" * 90)

from rfdetr import RFDETRBase

print(
    inspect.signature(
        RFDETRBase.train
    )
)


# ------------------------------------------------------------
# 7. Inspect RFDETRBase train source
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("RFDETRBase.train SOURCE")
print("=" * 90)

print(
    inspect.getsource(
        RFDETRBase.train
    )
)


# ------------------------------------------------------------
# 8. Load pretrained model
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("LOAD PRETRAINED MODEL")
print("=" * 90)

PRETRAINED_WEIGHTS = (
    "/home/jupyter/rf-detr-base-coco.pth"
)

model = RFDETRBase(
    pretrain_weights=PRETRAINED_WEIGHTS
)

print("✓ Model loaded")


# ------------------------------------------------------------
# 9. Get actual loaded model configuration
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("LOADED MODEL CONFIGURATION")
print("=" * 90)

if hasattr(model, "get_model_config"):

    print(
        inspect.signature(
            model.get_model_config
        )
    )

    try:
        loaded_config = (
            model.get_model_config()
        )

        print("\nReturned configuration:")
        print(loaded_config)

        if hasattr(
            loaded_config,
            "__dict__"
        ):

            print("\nConfiguration values:")

            for key, value in vars(
                loaded_config
            ).items():

                print(
                    f"{key:40} : {value}"
                )

    except Exception as e:

        print(
            "Could not call get_model_config():"
        )
        print(e)


# ------------------------------------------------------------
# 10. Inspect model class names
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("MODEL CLASS NAMES")
print("=" * 90)

if hasattr(model, "class_names"):

    print(
        model.class_names
    )

if hasattr(model.model, "class_names"):

    print(
        model.model.class_names
    )


# ------------------------------------------------------------
# 11. Inspect model arguments
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("MODEL ARGUMENTS")
print("=" * 90)

if hasattr(model.model, "args"):

    args = model.model.args

    print(type(args))

    if hasattr(args, "__dict__"):

        for key, value in vars(args).items():

            print(
                f"{key:40} : {value}"
            )

    else:

        print(args)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("CHECK COMPLETE")
print("=" * 90)

print(
    """
NO TRAINING WAS STARTED.
NO DATASET WAS MODIFIED.

Paste the output from this cell.
We will use the actual installed RF-DETR
implementation to rebuild the training pipeline.
"""
)
