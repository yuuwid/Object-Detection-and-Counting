import torch.backends.cudnn as cudnn
import torch
import cv2
import numpy as np
from pathlib import Path
import time
import shutil
import platform
import argparse
import sys
import os

from models.common import DetectMultiBackend

from utils.torch.device import select_device
from utils.general.directories import increment_path

from utils.general.checks import check_img_size, check_imshow
from utils.loader.stream import LoadStreams
from utils.loader.local import LoadImages, VID_FORMATS

from utils.general.nms import non_max_suppression
from utils.plot.annotator import Annotator
from utils.plot.color import colors

from utils.general.coordinate import xyxy2xywh
from utils.general.boxes import scale_boxes


from tracker.utils.parser import get_config
from tracker.deep_sort import DeepSort

from utils.torch.time import time_sync

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, "./yolov5")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

id_tracked = []

counted = {
    "mobil": 0,
    "truk": 0,
    "bis": 0,
    "motor": 0,
}


def detect(yolo, deepsort, config):
    source = config["source"]
    half = config["half"]

    webcam = (
        source == "0"
        or source.startswith("rtsp")
        or source.startswith("http")
        or source.endswith(".txt")
    )

    # Init Device (CPU / GPU)
    device = select_device(config["device"])
    half &= device.type != "cpu"

    exp_name = ""
    if type(yolo["model"]) is str:
        exp_name = yolo["model"].split(".")[0]

    exp_name = exp_name + "_" + deepsort["model"].split("/")[-1].split(".")[0]

    model = DetectMultiBackend(yolo["model"], device=device)
    stride, names = model.stride, model.names
    # check image size
    imgsz = check_img_size(yolo["img-size"], s=stride)
    # half precision only supported by PyTorch on CUDA
    half &= device.type != "cpu"
    model.model.half() if half else model.model.float()

    if config["show-vid"]:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataloader = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
        nr_sources = len(dataloader)
    else:
        dataloader = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        nr_sources = 1

    # initialize DEEPSORT
    cfg = get_config()
    cfg.merge_from_file(deepsort["config-deepsort"])

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deepsort["model"],
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE,
                n_init=cfg.DEEPSORT.N_INIT,
                nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run tracking
    model.warmup(imgsz=(1, 3, *imgsz))

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    for _, (path, im, im0s, _, s) in enumerate(dataloader):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference

        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred,
            yolo["conf-thres"],
            yolo["iou-thres"],
            config["classes"],
            config["agnostic-nms"],
            max_det=config["max-det"],
        )
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1

            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataloader.count
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataloader, "frame", 0)

            p = Path(p)

            # Anotator
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            w, h = im0.shape[1], im0.shape[0]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(
                    xywhs.cpu(),
                    confs.cpu(),
                    clss.cpu(),
                    im0,
                    True,
                )
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for _, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)

                        count_obj(bboxes, w, h, id, c, config["border-line"])

                        if show_vid:
                            # label = f"{id} {names[c]} {conf:.2f}"
                            label = f"{names[c]} {conf:.2f}"
                            annotator.box_label(bboxes, label, color=colors(c, True))

            else:
                # No detections
                deepsort_list[i].increment_ages()

            # Stream results
            im0 = annotator.result()

            if show_vid:
                color = (0, 0, 255)
                # start_point = (x_start, y_start)
                start_point = (
                    config["border-line"]["x_start"],
                    h - config["border-line"]["y_start"],
                )
                # end_point = (x_end, y_end)
                end_point = (
                    config["border-line"]["x_end"],
                    h - config["border-line"]["y_end"],
                )

                cv2.line(im0, start_point, end_point, color, thickness=2)

                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        # print(counted)


def count_obj(box, w, h, id, classid, borderline):
    global counted, id_tracked

    # box = [x y w h]

    center_coor = [
        int(box[0] + (box[2] - box[0]) / 2),
        int(box[1] + (box[3] - box[1]) / 2),
    ]

    padding_tolerance = borderline["padding-tolerance"]
    padding = ((box[3] - box[1]) / 2) * padding_tolerance

    if borderline["y_start"] == borderline["y_end"]:
        border_point = h - borderline["y_start"]

        if (
            center_coor[1] - padding < border_point
            and center_coor[1] + padding > border_point
            and id not in id_tracked
        ):
            if (
                center_coor[0] > borderline["x_start"]
                and center_coor[0] < borderline["x_end"]
            ):
                if classid == 0:
                    counted["bis"] += 1
                elif classid == 1:
                    counted["mobil"] += 1
                elif classid == 2:
                    counted["motor"] += 1
                elif classid == 3:
                    counted["truk"] += 1

                id_tracked.append(id)


def start():
    yolo = {
        "model": ROOT / "weights/yolov5-model.pt",
        "img-size": [640, 640],
        "conf-thres": 0.4,  # Input
        "iou-thres": 0.4,  # Input
    }

    deepsort = {
        "model": "weights/deep-sort-model.t7",
        "config-deepsort": "tracker/configs/deep_sort.yaml",
    }

    config = {
        "border-line": {
            "x_start": 200,  # Input
            "x_end": 1080,  # Input
            "y_start": 200,  # Input
            "y_end": 200,  # Input
            "padding-tolerance": 0.25,  # Input
        },
        "source": 'C:/Users/yuuwid/Videos/Video Skripsi/Delta-Pagi-20fps.m4v',
        "device": 0,
        "show-vid": True,
        "save-vid": False,
        "save-txt": False,
        "classes": None,
        "fourcc": "mp4v",
        "max-det": 20,  # Input
        "project": ROOT / "runs/track",
        "name": "exp",
        "exist-ok": False,
        "agnostic-nms": False,
        "augment": False,
        "update": False,
        "evaluate": False,
        "half": False,
        "visualize": False,
        "save-crop": False,
        "dnn": False,
    }

    detect(yolo, deepsort, config)


start()
