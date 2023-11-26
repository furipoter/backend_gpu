import subprocess
import uuid

import cv2
from flask import Blueprint, request, jsonify

from app import s3
from src.db.room import Room

import time

from models.experimental import attempt_load

import torch

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np


router = Blueprint('video', __name__, url_prefix='/video')


@router.route('/count/convert/<filename>')
def count_convert_file_name(filename):
    try:
        room = Room.query.filter_by(id=filename).first()
        if room is None:
            return jsonify({
                'message': 'Room not found'
            }), 404
        else:
            return jsonify({
                'count': len(room.number)
            })
    except Exception as e:
        return jsonify({
            'message': str(e)
        }), 500


# def update_room_number(filename):
#     try:
#         room_id = ".".join(filename.split('.')[:-1])
#         room_id = '-'.join(room_id.split('-')[:-1])
#         print("room_id", room_id)
#         # s3 에서 upload, convert 에서 개수 센 다음 작은걸로 업데이트
#         upload_count = len(s3.list_objects_v2(Bucket='furiosa-video', Prefix=f'upload/{room_id}')['Contents'])
#         convert_count = len(s3.list_objects_v2(Bucket='furiosa-video', Prefix=f'convert/{room_id}')['Contents'])
#         mini_number = min(upload_count, convert_count)
#
#         room = Room.query.filter_by(id=filename).first()
#         if room is None:
#             if mini_number == 1:
#                 return jsonify({
#                     'message': 'No video uploaded'
#                 }), 404
#             else:
#                 room = Room(id=room_id, number=mini_number)
#                 db.session.add(room)
#                 db.session.commit()
#                 return jsonify({
#                     'message': 'Room number updated successfully'
#                 })
#         else:
#             room.number = mini_number
#             db.session.commit()
#             return jsonify({
#                 'message': 'Room number updated successfully'
#             })
#     except Exception as e:
#         return jsonify({
#             'message': str(e)
#         }), 500


@router.route('/upload', methods=['POST'])
def video_upload():
    try:
        if 'video' in request.files:
            video = request.files['video']
            file_name = request.form['file_name']

            s3.upload_fileobj(video, 'furiosa-video', f'upload/{file_name}')
            upload_url = f'https://furiosa-video.s3.ap-northeast-2.amazonaws.com/upload/{file_name}'
            # update_room_number(file_name)
            # return jsonify({
            #     'message': 'Video uploaded successfully',
            #     'upload_url': upload_url
            # })
        else:
            return jsonify({'message': 'No video uploaded'})
        start_time = time.time()
        download_url = f'tmp/{file_name}'
        s3.download_file('furiosa-video', f'upload/{file_name}', download_url)
        # video = cv2.VideoCapture(download_url)
        video_no_audio_url = f'tmp/convert-{file_name}'

        face_blur(download_url, video_no_audio_url)
        print(time.time() - start_time)
        # s3에 convert 한 비디오 업로드
        s3.upload_fileobj(open(video_no_audio_url, 'rb'), 'furiosa-video', f'convert/{file_name}')
        convert_url = f'https://furiosa-video.s3.ap-northeast-2.amazonaws.com/convert/{file_name}'

        # update_room_number(file_name)
        return jsonify({
            'message': 'Video converted successfully',
            'convert_url': convert_url,
            'upload_url': upload_url,
        })
    except Exception as e:
        return jsonify({
            'message': str(e)
        }), 500


def face_blur(source, output):
    device='cuda'
    imgsz = 640
    model = attempt_load('yolov7-face_blur.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        # if device.type != 'cpu' and (
        #         old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        # old_img_b = img.shape[0]
        # old_img_h = img.shape[2]
        # old_img_w = img.shape[3]
        # for i in range(3):
        #     model(img, augment=False)[0]

        t1 = time_synchronized()
        # Inference
        pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=0,
                                   agnostic=False)
        t3 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            save_path = output
            # txt_path = str(save_dir / 'labels' / p.stem) + (
            #     '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            print(len(det))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # Add Object Blurring Code
                    # ..................................................................
                    crop_obj = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    blur = cv2.blur(crop_obj, (10, 10))
                    im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = blur
                    # ..................................................................

                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     if not hidedetarea:
                    #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #         print(f" The image with the result is saved in: {save_path}")
            #     else:  # 'video' or 'stream'
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
