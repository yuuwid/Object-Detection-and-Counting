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
count = 0
count2 = 0
data = []


# start_point = (x_start, y_start)
# end_point = (x_end, y_end)

coordinat_border = {"x_start": 320, "y_start": 400, "x_end": 240, "y_end": 400}


def detect(opt):
    (
        out,
        source,
        yolo_model,
        deep_sort_model,
        show_vid,
        save_vid,
        save_txt,
        imgsz,
        evaluate,
        half,
        project,
        exist_ok,
        update,
        save_crop,
    ) = (
        opt.output,
        opt.source,
        opt.yolo_model,
        opt.deep_sort_model,
        opt.show_vid,
        opt.save_vid,
        opt.save_txt,
        opt.imgsz,
        opt.evaluate,
        opt.half,
        opt.project,
        opt.exist_ok,
        opt.update,
        opt.save_crop,
    )

    webcam = (
        source == "0"
        or source.startswith("rtsp")
        or source.startswith("http")
        or source.endswith(".txt")
    )

    # Init Device (CPU / GPU)
    device = select_device(opt.device)

    # half precision only supported on CUDA
    half &= device.type != "cpu"

    output = opt.output
    evaluate = opt.evaluate

    # Output Folder
    if not evaluate:
        if os.path.exists(output):
            pass
            # delete output folder
            shutil.rmtree(output)
        os.makedirs(output)

    # Directories untuk single model
    exp_name = ""
    if type(opt.yolo_model) is str:
        exp_name = opt.yolo_model.split(".")[0]

    exp_name = exp_name + "_" + opt.deep_sort_model.split("/")[-1].split(".")[0]
    # Auto Named Save directory
    save_dir = increment_path(Path(opt.project) / exp_name, exist_ok=opt.exist_ok)

    # LOAD MODEL
    model = DetectMultiBackend(opt.yolo_model, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

    # Half
    half &= device.type != "cpu"  # half precision only supported by PyTorch on CUDA
    model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if opt.show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        nr_sources = 1
    vid_path, vid_writer, txt_path = (
        [None] * nr_sources,
        [None] * nr_sources,
        [None] * nr_sources,
    )

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                opt.deep_sort_model,
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
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = (
            increment_path(save_dir / Path(path[0]).stem, mkdir=True)
            if opt.visualize
            else False
        )
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            opt.classes,
            opt.agnostic_nms,
            max_det=opt.max_det,
        )
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f"{i}: "
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, "frame", 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = (
                        p.parent.name
                    )  # get folder name containing current img
                    # im.jpg, vid.mp4, ...
                    save_path = str(save_dir / p.parent.name)

            txt_path = str(save_dir / "tracks" / txt_file_name)  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0.copy() if opt.save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            w, h = im0.shape[1], im0.shape[0]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(
                    xywhs.cpu(), confs.cpu(), clss.cpu(), im0, True
                )
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        count_obj(bboxes, w, h, id)

                        if show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f"{id} {names[c]} {conf:.2f}"
                            annotator.box_label(bboxes, label, color=colors(c, True))

                # print(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                # print('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                color = (0, 0, 255)
                # start_point = (x_start, y_start)
                start_point = (
                    coordinat_border["x_start"],
                    h - coordinat_border["y_start"],
                )
                # end_point = (x_end, y_end)
                end_point = (
                    w - coordinat_border["x_end"],
                    h - coordinat_border["y_end"],
                )
                cv2.line(im0, start_point, end_point, color, thickness=2)
                org = (150, 150)
                font = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 3
                thickness = 3
                cv2.putText(
                    im0, str(count), org, font, fontScale, color, thickness, cv2.LINE_AA
                )

                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
    #     per image at shape {(1, 3, *imgsz)}' % t)


def count_obj(box, w, h, id):
    global count, count2, data

    # box = [x y w h]

    center_coor = [
        int(box[0] + (box[2] - box[0]) / 2),
        int(box[1] + (box[3] - box[1]) / 2),
    ]

    padding_tolerance = 0.25
    padding = ((box[3] - box[1]) / 2) * padding_tolerance

    if coordinat_border["y_start"] == coordinat_border["y_end"]:
        border_point = h - coordinat_border["y_start"]

        if (
            center_coor[1] - padding < border_point
            and center_coor[1] + padding > border_point
            and id not in data
        ):
            if (
                center_coor[0] > coordinat_border["x_start"]
                and center_coor[0] < w - coordinat_border["x_end"]
            ):
                count += 1
                data.append(id)
    else:
        # start_point = (x_start, y_start)
        start_point = (coordinat_border["x_start"], h - coordinat_border["y_start"])
        # end_point = (x_end, y_end)
        end_point = (w - coordinat_border["x_end"], h - coordinat_border["y_end"])

        steps_x = np.linspace(
            start_point[0], end_point[0], end_point[0] - start_point[0], dtype=int
        )
        steps_y = np.linspace(
            start_point[1], end_point[1], end_point[0] - start_point[0], dtype=int
        )

        if coordinat_border["x_start"] < coordinat_border["y_end"]:
            border_point_top = h - coordinat_border["y_start"]
            border_point_bottom = h - coordinat_border["y_end"]
        else:
            border_point_top = h - coordinat_border["y_end"]
            border_point_bottom = h - coordinat_border["y_start"]

        if (
            center_coor[1] - padding < border_point_top
            and center_coor[1] + padding > border_point_bottom
            and id not in data
        ):
            if (
                center_coor[0] > coordinat_border["x_start"]
                and center_coor[0] < w - coordinat_border["x_end"]
            ):
                count += 1
                data.append(id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_model",  #
        nargs="+",
        type=str,
        default=ROOT / "weights/yolov5-model.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(  #
        "--deep_sort_model", type=str, default="weights/deep-sort-model.t7"
    )
    parser.add_argument(  #
        "--source", type=str, default="video_test/cut3.avi", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(  #
        "--output", type=str, default="inference/output", help="output folder"
    )  # output folder
    parser.add_argument(  #
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(  #
        "--conf-thres", type=float, default=0.5, help="object confidence threshold"
    )
    parser.add_argument(  #
        "--iou-thres", type=float, default=0.5, help="IOU threshold for NMS"
    )
    parser.add_argument( #
        "--fourcc",
        type=str,
        default="mp4v",
        help="output video codec (verify ffmpeg support)",
    )
    parser.add_argument( #
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(  #
        "--show-vid", action="store_true", help="display tracking video results"
    )
    parser.add_argument(  #
        "--save-vid", action="store_true", help="save video tracking results"
    )
    parser.add_argument( #
        "--save-txt", action="store_true", help="save MOT compliant results to *.txt"
    )
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 16 17",
    )
    parser.add_argument( #
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference") #
    parser.add_argument("--update", action="store_true", help="update all models") #
    parser.add_argument("--evaluate", action="store_true", help="augmented inference") #
    parser.add_argument(  #
        "--config_deepsort", type=str, default="tracker/configs/deep_sort.yaml"
    )
    parser.add_argument( #
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument("--visualize", action="store_true", help="visualize features") #
    parser.add_argument( #
        "--max-det", type=int, default=1000, help="maximum detection per image"
    )
    parser.add_argument( #
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument( #
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(  #
        "--project", default=ROOT / "runs/track", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")  #
    parser.add_argument( #
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
