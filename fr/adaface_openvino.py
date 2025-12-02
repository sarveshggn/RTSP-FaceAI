# -*- coding: utf-8 -*-
# @Organization  : Adapted for OpenVINO
# @Author        : Sarvesh Adapt
# @Time          : 2025-07-02
# @Function      : AdaFace OpenVINO inference

import numpy as np
import cv2
from openvino.runtime import Core
import fd.face_align as face_align

__all__ = [
    'AdaFaceOpenVINO',
]

class AdaFaceOpenVINO:
    def __init__(self, model_file=None):
        assert model_file is not None
        self.model_file = model_file
        self.core = Core()

        # Load IR model
        self.compiled_model = self.core.compile_model(model_file, "NPU")
        self.input_tensor = self.compiled_model.inputs[0]
        self.output_tensor = self.compiled_model.outputs[0]

        # Assumes single input
        input_shape = self.input_tensor.shape
        self.input_size = tuple(input_shape[2:4][::-1])  # (W,H)
        self.input_shape = input_shape

        # Set normalization to match AdaFace ([-1, 1] range)
        self.input_mean = 127.5
        self.input_std = 127.5

        print(f"Loaded OpenVINO model: {model_file}")
        print(f"Input shape: {self.input_shape}, using primary output.")

    def prepare(self, ctx_id, **kwargs):
        # For API compatibility — no-op in OpenVINO
        pass

    def get(self, img, kps):
        aimg = face_align.norm_crop(img, landmark=kps, image_size=self.input_size[0])
        embedding = self.get_feat(aimg).flatten()
        return embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]

        input_size = self.input_size

        # Prepare blob, scale to [-1, 1], BGR order
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean),
                                      swapRB=False)

        # Run inference
        result = self.compiled_model([blob])[self.output_tensor]

        return result

    def forward(self, batch_data):
        # batch_data assumed already scaled appropriately to [-1,1]
        result = self.compiled_model([batch_data])[self.output_tensor]
        return result
