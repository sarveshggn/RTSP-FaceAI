from __future__ import division
import datetime
import numpy as np
import openvino as ov
import os
import os.path as osp
import cv2
import sys

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    def __init__(self, model_file=None, compiled_model=None, blur_threshold=50.0):
        self.model_file = model_file
        self.compiled_model = compiled_model
        self.taskname = 'detection'
        self.batched = False
        self.blur_threshold = blur_threshold  # Laplacian variance threshold for blur detection
        
        if self.compiled_model is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            
            # Initialize OpenVINO Core
            self.core = ov.Core()
            
            # Load and compile the model
            self.model = self.core.read_model(self.model_file)
            # Use GPU if available, otherwise CPU
            available_devices = self.core.available_devices
            device = "GPU" # if "GPU" in available_devices else "CPU"
            self.compiled_model = self.core.compile_model(self.model, device)
        
        self.center_cache = {}
        self.nms_thresh = 0.5
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        # Get input configuration
        input_layer = self.compiled_model.input(0)
        
        # Handle dynamic shapes
        try:
            input_shape = input_layer.shape
            # Check if shape is dynamic
            if input_layer.partial_shape.is_dynamic:
                # For your specific model: 1, 3, 360, 640 (NCHW format)
                self.input_size = (640, 360)  # (width, height)
                input_shape = [1, 3, 360, 640]
            else:
                input_shape = list(input_shape)
                # Extract height and width from NCHW format
                if len(input_shape) >= 4:
                    height, width = input_shape[2], input_shape[3]
                    self.input_size = (width, height)  # (width, height)
        except RuntimeError:
            # If shape is dynamic, use your model's specific dimensions
            self.input_size = (640, 360)  # (width, height)
            input_shape = [1, 3, 360, 640]
        
        self.input_name = input_layer.get_any_name()
        self.input_shape = input_shape
        
        # Get output configuration
        outputs = self.compiled_model.outputs
        
        # Handle dynamic output shapes
        try:
            first_output_shape = outputs[0].shape
            if len(first_output_shape) == 3:
                self.batched = True
        except RuntimeError:
            # If output shape is dynamic, assume non-batched initially
            self.batched = False
        
        self.output_names = [output.get_any_name() for output in outputs]
        self.input_mean = 127.5
        self.input_std = 128.0
        
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def is_blurred(self, face_img):
        """
        Check if a face image is blurred using Laplacian variance method.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            bool: True if face is blurred (should be filtered out), False otherwise
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Return True if blurred (variance below threshold)
        return laplacian_var < self.blur_threshold

    def filter_blurred_faces(self, img, detections, kpss=None):
        """
        Filter out blurred faces from detection results.
        
        Args:
            img: Original image
            detections: Detection results array (N, 5) where each row is [x1, y1, x2, y2, score]
            kpss: Keypoints array (optional)
            
        Returns:
            tuple: (filtered_detections, filtered_kpss)
        """
        if len(detections) == 0:
            return detections, kpss
        
        filtered_indices = []
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det[:4].astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            # Skip if bounding box is invalid
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Extract face region
            face_img = img[y1:y2, x1:x2]
            
            # Skip if face is too small
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue
            
            # Check if face is blurred
            if not self.is_blurred(face_img):
                filtered_indices.append(i)
        
        # Filter detections and keypoints
        if len(filtered_indices) > 0:
            filtered_detections = detections[filtered_indices]
            filtered_kpss = kpss[filtered_indices] if kpss is not None else None
        else:
            filtered_detections = np.array([]).reshape(0, 5)
            filtered_kpss = None
            
        return filtered_detections, filtered_kpss

    def prepare(self, ctx_id, **kwargs):
        # OpenVINO doesn't have the same device switching mechanism as ONNX
        # Device is set during compilation
        if ctx_id < 0:
            # Re-compile for CPU if needed
            if hasattr(self, 'model'):
                self.compiled_model = self.core.compile_model(self.model, "CPU")
        
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size
        blur_threshold = kwargs.get('blur_threshold', None)
        if blur_threshold is not None:
            self.blur_threshold = blur_threshold

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, 
                                   (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        
        # Create input tensor with proper shape
        input_tensor = ov.Tensor(array=blob)
        
        # Run inference with OpenVINO
        results = self.compiled_model({self.input_name: input_tensor})
        
        # Convert results to list format similar to ONNX
        net_outs = []
        for output in self.compiled_model.outputs:
            output_name = output.get_any_name()
            net_outs.append(results[output_name])
        
        # Dynamically determine if model is batched from actual output shapes
        if not hasattr(self, '_batched_determined'):
            if len(net_outs) > 0 and len(net_outs[0].shape) == 3:
                self.batched = True
            else:
                self.batched = False
            self._batched_determined = True

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            # Reshape predictions to match anchor_centers
            if self.batched:
                # For batched models, reshape to (N, C) where N matches anchor_centers
                scores = scores.reshape(-1)
                bbox_preds = bbox_preds.reshape(-1, 4)
                if self.use_kps:
                    kps_preds = kps_preds.reshape(-1, 10)
            else:
                # For non-batched models, ensure proper reshaping
                if len(scores.shape) == 3:
                    scores = scores.reshape(-1)
                if len(bbox_preds.shape) == 3:
                    bbox_preds = bbox_preds.reshape(-1, 4)
                if self.use_kps and len(kps_preds.shape) == 3:
                    kps_preds = kps_preds.reshape(-1, 10)
            
            # Ensure shapes match
            expected_size = anchor_centers.shape[0]
            if scores.shape[0] != expected_size:
                # Truncate or pad to match expected size
                if scores.shape[0] > expected_size:
                    scores = scores[:expected_size]
                    bbox_preds = bbox_preds[:expected_size]
                    if self.use_kps:
                        kps_preds = kps_preds[:expected_size]
                else:
                    # This shouldn't happen normally, but handle gracefully
                    print(f"Warning: Prediction size {scores.shape[0]} < expected {expected_size}")
                    # Pad with zeros if needed
                    pad_size = expected_size - scores.shape[0]
                    scores = np.pad(scores, (0, pad_size), 'constant', constant_values=0)
                    bbox_preds = np.pad(bbox_preds, ((0, pad_size), (0, 0)), 'constant', constant_values=0)
                    if self.use_kps:
                        kps_preds = np.pad(kps_preds, ((0, pad_size), (0, 0)), 'constant', constant_values=0)

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, thresh=None, max_num=0, metric='default', filter_blur=True):
        # Use model's fixed input size if not specified
        if input_size is None:
            input_size = self.input_size if self.input_size is not None else (640, 360)
        
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]  # height/width
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # Create detection image with model's input dimensions
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)  # (height, width, channels)
        det_img[:new_height, :new_width, :] = resized_img
        
        det_thresh = thresh if thresh is not None else self.det_thresh

        scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        
        # Filter blurred faces if enabled
        if filter_blur:
            det, kpss = self.filter_blurred_faces(img, det, kpss)
        
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def autodetect(self, img, max_num=0, metric='max', filter_blur=True):
        # Simple single detection pass with model's fixed input size
        return self.detect(img, input_size=self.input_size, thresh=0.5, max_num=max_num, metric=metric, filter_blur=filter_blur)

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
