from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np
import io


def main(visulization=False):

    dataset_path = 'D:/A405/project/benchmark/CST-AntiUAV-main/CST-AntiUAV'

    # test or val
    #subset = 'test'
    # subset = 'val'
    subset = 'test'
    # show window size
    winwidth=640
    winheight=512

    #seq_name = 'all'
    seq_name = 'urban-areas_48'
    # seq_name = 'building_47'
    # seq_name = 'cn_mountains_17'
    save_path='D:/A405/project/benchmark/visual_CST'

    dataset_path = os.path.join(dataset_path, subset)

    # mode 1 means the results is formatted by (x,y,w,h)
    # mode 2 means the results is formatted by (x1,y1,x2,y2)


    label_files = sorted(glob.glob(
        os.path.join(dataset_path, '*/IR_label.json')))

    if visulization:
        cv2.namedWindow("Tracking", 0)
        cv2.resizeWindow("Tracking", winwidth, winheight)

    for video_id, label_file in enumerate(label_files, start=1):

        # groundtruth
        with open(label_file, 'r') as f:
            label_res = json.load(f)

        video_dirs = os.path.dirname(label_file)
        video_dirsbase = os.path.basename(video_dirs)

        if seq_name == 'all':
            pass
        elif video_dirsbase == seq_name:
            pass
        else:
            continue

        image_path = os.path.join(save_path, video_dirsbase)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # 创建原图和加框图像的文件夹
        original_image_path = os.path.join(image_path, '原图')
        annotated_image_path = os.path.join(image_path, '加框图')

        if not os.path.exists(original_image_path):
            os.makedirs(original_image_path)
        if not os.path.exists(annotated_image_path):
            os.makedirs(annotated_image_path)

        img_files = sorted(glob.glob(
            os.path.join(dataset_path, video_dirsbase, '*.jpg')))

        for frame_id, img_file in enumerate(img_files):

            frame = cv2.imread(img_file)

            # 保存原图
            original_image_file = os.path.join(original_image_path, str(frame_id + 1).zfill(4) + '.png')
            cv2.imwrite(original_image_file, frame)

            # 加框处理
            _gt = label_res['gt'][frame_id]
            _exist = label_res['exist'][frame_id]
            if _exist:
                cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                              (0, 0, 255))  # (0,255,0) red
            # cv2.putText(frame, 'exist' if _exist else 'not exist',
            #             (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)
            # cv2.putText(frame, video_dirsbase, (frame.shape[1] - 225, frame.shape[0]-10), 1, 1, (255, 255, 0), 2)

            if visulization:
                cv2.resizeWindow("Tracking", winwidth, winheight)
                cv2.imshow("Tracking", frame)
                cv2.waitKey(10)

            # 保存加框后的图像
            annotated_image_file = os.path.join(annotated_image_path, str(frame_id + 1).zfill(4) + '.png')
            cv2.imwrite(annotated_image_file, frame)

    cv2.destroyAllWindows()




if __name__ == '__main__':
    main(visulization=True)
