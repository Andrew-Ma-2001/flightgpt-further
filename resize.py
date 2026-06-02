import argparse
import json
from pathlib import Path

import cv2


TARGET_SIZE = (2048, 2048)


def resize_jpg_tree(input_dir: Path, output_dir: Path) -> None:
    """
    Recursively find JPG images under input_dir, resize them to TARGET_SIZE
    using bicubic interpolation, and save them under output_dir while
    preserving the relative directory structure.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    jpg_files = [
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg"
    ]

    for src_path in jpg_files:
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(src_path))
        if image is None:
            print(f"[WARN] Failed to read image, skipping: {src_path}")
            continue

        resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        ok = cv2.imwrite(str(dst_path), resized)
        if not ok:
            print(f"[WARN] Failed to write image: {dst_path}")


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _scale_point(point: list, scale_x: float, scale_y: float) -> list:
    x = int(round(point[0] * scale_x))
    y = int(round(point[1] * scale_y))
    return [_clamp(x, 0, TARGET_SIZE[0] - 1), _clamp(y, 0, TARGET_SIZE[1] - 1)]


def _scale_bbox(bbox: list, scale_x: float, scale_y: float) -> list:
    x1 = int(round(bbox[0] * scale_x))
    y1 = int(round(bbox[1] * scale_y))
    x2 = int(round(bbox[2] * scale_x))
    y2 = int(round(bbox[3] * scale_y))
    return [
        _clamp(x1, 0, TARGET_SIZE[0]),
        _clamp(y1, 0, TARGET_SIZE[1]),
        _clamp(x2, 0, TARGET_SIZE[0]),
        _clamp(y2, 0, TARGET_SIZE[1]),
    ]


def resize_training_json(input_json: Path, output_json: Path) -> None:
    input_json = input_json.resolve()
    output_json = output_json.resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Input json does not exist: {input_json}")

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Training json must be a list of records.")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[WARN] Record #{idx} is not an object, skipping.")
            continue

        image_size = item.get("image_size")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or image_size[0] == 0
            or image_size[1] == 0
        ):
            print(f"[WARN] Record #{idx} has invalid image_size, skipping scaling.")
            continue

        src_w, src_h = image_size
        scale_x = TARGET_SIZE[0] / float(src_w)
        scale_y = TARGET_SIZE[1] / float(src_h)

        if isinstance(item.get("start_position"), list) and len(item["start_position"]) == 2:
            item["start_position"] = _scale_point(item["start_position"], scale_x, scale_y)

        if isinstance(item.get("target_position"), list) and len(item["target_position"]) == 2:
            item["target_position"] = _scale_point(item["target_position"], scale_x, scale_y)

        landmark_bbox = item.get("landmark_bbox")
        if isinstance(landmark_bbox, list):
            scaled_bbox = []
            for box in landmark_bbox:
                if isinstance(box, list) and len(box) == 4:
                    scaled_bbox.append(_scale_bbox(box, scale_x, scale_y))
                else:
                    scaled_bbox.append(box)
            item["landmark_bbox"] = scaled_bbox

        item["image_size"] = [TARGET_SIZE[0], TARGET_SIZE[1]]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize all JPG images in a directory tree to 2048x2048, and optionally "
            "scale training-json annotations to match the new image size."
        )
    )
    parser.add_argument("input_path", nargs="?", type=Path, help="Input root directory")
    parser.add_argument("output_path", nargs="?", type=Path, help="Output root directory")
    parser.add_argument("--input-json", type=Path, help="Input training json path")
    parser.add_argument("--output-json", type=Path, help="Output training json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_image_resize = args.input_path is not None or args.output_path is not None
    run_json_resize = args.input_json is not None or args.output_json is not None

    if run_image_resize:
        if args.input_path is None or args.output_path is None:
            raise ValueError("Both input_path and output_path are required for image resize.")
        resize_jpg_tree(args.input_path, args.output_path)

    if run_json_resize:
        if args.input_json is None or args.output_json is None:
            raise ValueError("Both --input-json and --output-json are required.")
        resize_training_json(args.input_json, args.output_json)

    if not run_image_resize and not run_json_resize:
        raise ValueError(
            "No task selected. Provide input/output paths for images and/or "
            "--input-json/--output-json for json scaling."
        )


if __name__ == "__main__":
    main()
