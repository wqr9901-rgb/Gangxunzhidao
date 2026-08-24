# test_qc_engine.py  -- 质检规则引擎单元测试
import json
import unittest

from qc_engine import (
    parse_labelme, parse_coco, parse_yolo,
    load_annotation, load_annotation_with_img, check_boxes
)


class TestParse(unittest.TestCase):
    def test_labelme(self):
        data = {"imageWidth": 500, "imageHeight": 400, "shapes": [
            {"label": "car", "shape_type": "rectangle", "points": [[10, 10], [110, 90]]}
        ]}
        boxes, w, h = parse_labelme(data)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["w"], 100)
        self.assertEqual(boxes[0]["h"], 80)
        self.assertEqual((w, h), (500, 400))

    def test_coco(self):
        data = {
            "images": [{"id": 1, "width": 500, "height": 400}],
            "categories": [{"id": 1, "name": "dog"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 200, 150]}]
        }
        boxes, w, h = parse_coco(data)
        self.assertEqual(boxes[0]["label"], "dog")
        self.assertEqual((w, h), (500, 400))

    def test_yolo(self):
        boxes, w, h = parse_yolo("0 0.5 0.5 0.2 0.3", 500, 400)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["w"], 100)
        self.assertAlmostEqual(boxes[0]["h"], 120)


class TestLoad(unittest.TestCase):
    def test_load_labelme_json(self):
        data = {"imageWidth": 500, "imageHeight": 400, "shapes": [
            {"label": "a", "shape_type": "rectangle", "points": [[0, 0], [50, 50]]}
        ]}
        r = load_annotation("ann.json", json.dumps(data).encode())
        self.assertTrue(r["ok"])
        self.assertEqual(r["format"], "Labelme")

    def test_load_coco_json(self):
        data = {"images": [{"id": 1, "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "a"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10]}]}
        r = load_annotation("ann.json", json.dumps(data).encode())
        self.assertEqual(r["format"], "COCO")

    def test_load_invalid_json(self):
        r = load_annotation("ann.json", b"{bad json")
        self.assertFalse(r["ok"])

    def test_load_unknown_structure(self):
        r = load_annotation("ann.json", b'{"foo": 1}')
        self.assertFalse(r["ok"])

    def test_yolo_requires_image(self):
        r = load_annotation("ann.txt", b"0 0.5 0.5 0.2 0.3")
        self.assertFalse(r["ok"])

    def test_yolo_with_image(self):
        r = load_annotation_with_img("ann.txt", b"0 0.5 0.5 0.2 0.3", 500, 400)
        self.assertTrue(r["ok"])
        self.assertEqual(r["format"], "YOLO")


class TestCheck(unittest.TestCase):
    def test_overflow_detected(self):
        boxes = [{"label": "car", "x": -5, "y": 10, "w": 100, "h": 80}]
        res = check_boxes(boxes, 500, 400)
        self.assertEqual(res["checks"]["框是否越界"]["status"], "fail")
        self.assertTrue(res["checks"]["框是否越界"]["evidence"])

    def test_tiny_box_detected(self):
        boxes = [{"label": "x", "x": 10, "y": 10, "w": 15, "h": 15}]
        res = check_boxes(boxes, 500, 400)
        self.assertEqual(res["checks"]["框是否过小"]["status"], "fail")

    def test_normal_box_pass(self):
        boxes = [{"label": "x", "x": 10, "y": 10, "w": 200, "h": 150}]
        res = check_boxes(boxes, 500, 400)
        self.assertEqual(res["checks"]["框是否越界"]["status"], "pass")
        self.assertEqual(res["checks"]["框是否过小"]["status"], "pass")
        self.assertEqual(res["summary"], "全部通过")

    def test_empty_annotation_warn(self):
        res = check_boxes([], 500, 400)
        self.assertEqual(res["checks"]["是否存在明显漏标"]["status"], "warn")

    def test_no_image_size(self):
        res = check_boxes([{"label": "x", "x": 0, "y": 0, "w": 10, "h": 10}], 0, 0)
        self.assertFalse(res["ok"])


if __name__ == "__main__":
    unittest.main()
