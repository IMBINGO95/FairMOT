import numpy as np
import cv2


def draw_detection_1(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    new_bbox_det = []
    new_bbox_det = np.array(new_bbox_det)
    for i, box in enumerate(bboxes):
        area = (box[0] - box[2]) * (box[1] - box[3])
        if (area > 500*300) or (scores[i] < thr) or (int(cls_inds[i]) != 1):
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 1000)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-4 * h, colors[cls_indx], thick // 3)

        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)

        sub_new_det = np.array((box[0] , box[1] , box[2] , box[3], cls_indx))
        new_bbox_det = np.insert(new_bbox_det, 0, values=sub_new_det, axis=0)
        length = new_bbox_det.size / 5

    new_bbox_det.resize(int(length),5)
    return imgcv,new_bbox_det